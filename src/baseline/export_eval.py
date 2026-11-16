from __future__ import annotations

import argparse
import hashlib
import copy
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from BEATs import BEATs, BEATsConfig
from src.baseline.baseline_metrics import (
    ReferenceBundle,
    compute_baseline_metrics,
    correct_logit_margin,
    top1_top2_gap,
)
from src.baseline.manifest import (
    attach_run_manifest_hash,
    canonical_json_hash,
    file_sha256,
    git_commit,
    write_run_manifest,
    write_run_manifests,
)
from src.baseline.quant_checkpoint import collect_qparams_from_controller, load_qparams_csv, qparams_hash, qparams_to_jsonable
from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
from src.metrics.metric_config import DEFAULT_EPS, DEFAULT_SCALAR_OER_TOP_RATIO
from src.quant.ffn_quant_injector import apply_fixed_qparams_to_quant_module
from src.quant.quant_core import iter_quantized_linears, prepare_beats_ffn_int4_quantization_from_project_config
from src.utils.group_utils import group_protocol_metadata, read_canonical_bucket, tail_included, validate_bucket_values
from src.utils.token_trace_hook import TokenTraceHookManager, resolve_full_layer_indices


DEFAULT_TOKEN_LAYERS = ",".join(str(index) for index in range(12))
DEFAULT_SPARSE_TOKEN_LAYERS = "0,3,6,9,11"
DEFAULT_TOKEN_EXPORT = "all"
DEFAULT_MAX_EXPORT_PAYLOAD_GIB = 16.0
DEFAULT_POOLING = "mask_aware_mean"
POOLING_DEFINITION = "mask_aware_mean(final_encoder_output_excluding_padding)"
REFERENCE_MODEL_NAME = "fp32_teacher"
REFERENCE_PROTOCOL_VERSION = "baseline_fp32_reference.v1"
OER_PROTOCOL_VERSION = "baseline_oer_oracle.v1"
OER_ORACLE_FILENAME = "fp32_oer_oracle.pt"
OER_ORACLE_MANIFEST_FILENAME = "fp32_oer_oracle.manifest.json"
OER_ACTIVATION_POSITION_USED = "input"
OER_FFN_MODULE_NAMES = ("fc1", "fc2")
OER_EXPECTED_NUM_LAYERS = 12
OER_EPS = float(DEFAULT_EPS)
OER_TOPK_RATIO = float(DEFAULT_SCALAR_OER_TOP_RATIO)


@dataclass(frozen=True)
class ExportEvalConfig:
    model_name: str
    checkpoint_path: str | Path
    split_csv: str | Path
    split_name: str
    output_dir: str | Path
    label_mapping_path: str | Path
    bucket_protocol: Any
    pooling: str = DEFAULT_POOLING
    token_export: str = DEFAULT_TOKEN_EXPORT
    token_layers: Sequence[int] | str = DEFAULT_TOKEN_LAYERS
    device: str | torch.device | None = None
    seed: int = 2026
    dataset_root: str | Path | None = None
    batch_size: int = 8
    num_workers: int = 0
    amp: bool = False
    export_all_token_layers: bool = False
    max_export_payload_gib: float = DEFAULT_MAX_EXPORT_PAYLOAD_GIB
    calibration_hash: str | None = None
    quantized_layers: Sequence[Mapping[str, Any]] | None = None
    bitwidth: str | Mapping[str, Any] = "fp32"
    reference_root: str | Path | None = None
    qparams_csv_override: str | Path | None = None


@dataclass(frozen=True)
class _SingleSplitPaths:
    name: str
    root: Path | None
    train_csv: Path
    eval_csv: Path
    test_csv: Path
    export_csv: Path
    label_mapping: Path
    output_checkpoint: Path



def audio_config_from_project() -> AudioInputConfig:
    audio = getattr(project_config, "AUDIO", {})
    if not isinstance(audio, Mapping):
        audio = {}
    return AudioInputConfig(
        sample_rate=int(audio.get("TARGET_SR", 16000)),
        target_seconds=float(audio.get("TARGET_SECONDS", 10.0)),
        normalization=str(audio.get("NORMALIZATION", "peak")),
        eps=float(audio.get("EPS", 1e-8)),
        deterministic_crop=str(audio.get("CROP_MODE", "center")),
        fixed_crop_start=int(audio.get("FIXED_CROP_START", 0)),
    )


def _validate_dense_label_mapping(label_mapping: Mapping[str, int], *, context: str) -> list[int]:
    if not isinstance(label_mapping, Mapping) or not label_mapping:
        raise ValueError(f"{context} must be a non-empty mapping")
    ids = [int(value) for value in label_mapping.values()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{context} label_id values must be unique, got duplicates in {sorted(ids)}")
    if any(label_id < 0 for label_id in ids):
        raise ValueError(f"{context} label_id values must be non-negative, got {sorted(ids)}")
    expected = list(range(len(ids)))
    observed = sorted(ids)
    if observed != expected:
        raise ValueError(f"{context} label_id values must be dense 0..C-1; observed={observed}, expected={expected}")
    return observed


def num_predictor_classes(label_mapping: Mapping[str, int]) -> int:
    return len(_validate_dense_label_mapping(label_mapping, context="label_mapping"))


def _is_missing_csv_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "<na>"}


def _load_label_mapping_json(path: str | Path) -> dict[str, int]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"label_mapping_path not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"label_mapping_path must contain a non-empty JSON mapping: {target}")
    result: dict[str, int] = {}
    for label_name, label_id in raw.items():
        value = int(label_id)
        if value < 0:
            raise ValueError(f"label_mapping has negative label_id for {label_name!r}: {value}")
        result[str(label_name)] = value
    _validate_dense_label_mapping(result, context=f"label_mapping_path {target}")
    return result


def _canonical_bucket(value: Any, *, context: str) -> str:
    return validate_bucket_values([value], context=context)[0]


def _label_buckets_from_mapping(bucket_protocol: Mapping[Any, Any]) -> dict[int, str]:
    raw: Any = bucket_protocol
    for key in ("label_buckets", "bucket_by_label_id", "buckets_by_label_id"):
        if isinstance(raw, Mapping) and key in raw:
            raw = raw[key]
            break
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
        result: dict[int, str] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"bucket_protocol sequence item {index} must be a mapping")
            label_id = int(item.get("label_id"))
            bucket_value = item.get("bucket", item.get("shot_group"))
            result[label_id] = _canonical_bucket(bucket_value, context=f"bucket_protocol[{index}]")
        return result
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("bucket_protocol mapping must provide label_id -> bucket entries")
    result: dict[int, str] = {}
    for label_id, bucket_value in raw.items():
        result[int(label_id)] = _canonical_bucket(bucket_value, context=f"bucket_protocol.label_id={label_id}")
    return result


def _label_buckets_from_csv(path: str | Path) -> dict[int, str]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"bucket_protocol CSV not found: {target}")
    buckets_by_label: dict[int, set[str]] = {}
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label_id" not in reader.fieldnames:
            raise KeyError(f"bucket_protocol CSV must contain label_id column: {target}")
        for row_number, row in enumerate(reader, start=2):
            if _is_missing_csv_value(row.get("label_id")):
                continue
            label_id = int(float(str(row.get("label_id")).strip()))
            bucket_value = row.get("bucket")
            shot_group_value = row.get("shot_group")
            train_label_n_value = row.get("train_label_n", row.get("label_n"))
            result = read_canonical_bucket(
                bucket=None if _is_missing_csv_value(bucket_value) else [bucket_value],
                shot_group=None if _is_missing_csv_value(shot_group_value) else [shot_group_value],
                train_label_n=None if _is_missing_csv_value(train_label_n_value) else [train_label_n_value],
                context=f"{target}:row={row_number}",
                allow_shot_group_fallback=True,
            )
            buckets_by_label.setdefault(label_id, set()).add(result.bucket[0])
    conflicts = {label_id: sorted(values) for label_id, values in buckets_by_label.items() if len(values) > 1}
    if conflicts:
        preview = "; ".join(f"label_id={label_id}: buckets={values}" for label_id, values in sorted(conflicts.items())[:8])
        raise ValueError(f"{target}: inconsistent bucket metadata for the same label_id; {preview}")
    return {int(label_id): sorted(values)[0] for label_id, values in buckets_by_label.items()}


def _resolve_bucket_protocol(bucket_protocol: Any, *, split_csv: str | Path) -> dict[int, str]:
    if isinstance(bucket_protocol, Mapping):
        return _label_buckets_from_mapping(bucket_protocol)
    if isinstance(bucket_protocol, (str, Path)):
        text = str(bucket_protocol).strip()
        if text.lower() in {"split_csv", "split"}:
            return _label_buckets_from_csv(split_csv)
        target = Path(text)
        if target.suffix.lower() == ".json":
            with target.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, Mapping):
                raise ValueError(f"bucket_protocol JSON must contain a mapping: {target}")
            return _label_buckets_from_mapping(raw)
        return _label_buckets_from_csv(target)
    raise TypeError("bucket_protocol must be a mapping, JSON/CSV path, or 'split_csv'")


def _flat_export_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "logits": root,
        "embeddings": root,
        "features": root,
        "predictions": root,
        "metrics": root,
        "manifests": root,
    }


def _export_arg_namespace(config: ExportEvalConfig) -> argparse.Namespace:
    if bool(config.amp):
        raise ValueError("export_eval formal export forbids AMP; run deterministic FP32 inference only")
    token_layers = config.token_layers
    if isinstance(token_layers, Sequence) and not isinstance(token_layers, (str, bytes)):
        token_layers = ",".join(str(int(item)) for item in token_layers)
    return argparse.Namespace(
        eval_batch_size=int(config.batch_size),
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        seed=int(config.seed),
        amp=False,
        token_export=str(config.token_export),
        export_all_token_layers=bool(config.export_all_token_layers),
        token_layers=str(token_layers),
        max_export_payload_gib=float(config.max_export_payload_gib),
        export_name=str(config.model_name),
        flat_output=True,
        run_identity_hash=None,
        reference_root=None if config.reference_root is None else str(Path(config.reference_root)),
    )


def build_loader(paths: Any, *, split: str, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    if split == "eval":
        split_csv = Path(paths.eval_csv)
        mode = "eval"
    elif split == "test":
        split_csv = Path(paths.test_csv)
        mode = "test"
    elif split == "export":
        split_csv_value = getattr(paths, "export_csv", None)
        if split_csv_value is None:
            raise ValueError("export split requires paths.export_csv")
        split_csv = Path(split_csv_value)
        mode = "export"
    else:
        raise ValueError(f"export_eval only supports eval/test/export splits, got {split!r}")

    dataset = AudioInputDataset(
        split_csv,
        mode=mode,
        dataset_root=None if getattr(paths, "root", None) is None else Path(paths.root),
        project_root=PROJECT_ROOT,
        audio_config=audio_config_from_project(),
        global_seed=int(seed),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
    )


def _split_csv_path(paths: Any, split: str) -> Path:
    if split == "eval":
        return Path(paths.eval_csv)
    if split == "test":
        return Path(paths.test_csv)
    if split == "export":
        split_csv_value = getattr(paths, "export_csv", None)
        if split_csv_value is None:
            raise ValueError("export split requires paths.export_csv")
        return Path(split_csv_value)
    raise ValueError(f"export_eval only supports eval/test/export splits, got {split!r}")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value.to(device=device, non_blocking=True)
        else:
            result[key] = value
    return result


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_token_layers(value: str | None) -> tuple[int, ...]:
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    layers: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            layer_idx = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid layer index in --token-layers: {item!r}") from exc
        if layer_idx < 0:
            raise argparse.ArgumentTypeError(f"layer index must be non-negative, got {layer_idx}")
        layers.append(layer_idx)
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError(f"--token-layers contains duplicates: {layers}")
    return tuple(layers)


def _resolve_export_token_layers(model: nn.Module, args: argparse.Namespace) -> tuple[int, ...]:
    all_layers = resolve_full_layer_indices(model)
    token_export = str(getattr(args, "token_export", "")).strip().lower()
    if not token_export:
        token_export = "all"
    if bool(getattr(args, "export_all_token_layers", False)):
        token_export = "all"
    if token_export == "none":
        return ()
    if token_export == "all":
        return all_layers
    if token_export != "sparse":
        raise ValueError(f"token_export must be all/sparse/none, got {token_export!r}")
    raw_token_layers = getattr(args, "token_layers", "")
    if str(raw_token_layers).strip() == DEFAULT_TOKEN_LAYERS:
        raw_token_layers = DEFAULT_SPARSE_TOKEN_LAYERS
    selected = _parse_token_layers(str(raw_token_layers))
    if not selected:
        raise ValueError("token_export='sparse' requires a non-empty --token-layers list")
    invalid = [item for item in selected if item not in set(all_layers)]
    if invalid:
        raise ValueError(f"token layer indices out of range for model layers {all_layers}: {invalid}")
    return selected


def _ensure_token_padding_mask(
    padding_mask: Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> Tensor:
    if padding_mask is None:
        raise RuntimeError("export_eval requires padding_mask in every batch; refusing fail-open pooling")
    mask = padding_mask.to(device=device, dtype=torch.bool)
    if tuple(mask.shape) != (int(batch_size), int(num_tokens)):
        raise ValueError(
            f"token padding_mask must be [B,T]={int(batch_size), int(num_tokens)}, got {tuple(mask.shape)}"
        )
    return mask.contiguous()


def prepare_encoder_input_for_export(
    model: BEATs,
    waveform: Tensor,
    padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor]:
    if padding_mask is None:
        raise KeyError("batch missing padding_mask; export/calibration/KD pooling requires explicit padding metadata")
    fbank = model.preprocess(waveform.float())
    feature_padding_mask = model.forward_padding_mask(fbank, padding_mask.to(device=fbank.device, dtype=torch.bool))
    fbank = fbank.unsqueeze(1)
    features = model.patch_embedding(fbank)
    features = features.reshape(features.shape[0], features.shape[1], -1)
    features = features.transpose(1, 2)
    features = model.layer_norm(features)
    feature_padding_mask = model.forward_padding_mask(features, feature_padding_mask)
    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)
    encoder_input = model.dropout_input(features)
    token_padding_mask = _ensure_token_padding_mask(
        feature_padding_mask,
        batch_size=int(encoder_input.shape[0]),
        num_tokens=int(encoder_input.shape[1]),
        device=encoder_input.device,
    )
    return encoder_input, feature_padding_mask, token_padding_mask


def mask_aware_mean(features: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
    features = features.float()
    if features.ndim != 3:
        raise ValueError(f"features must be [B,T,D], got shape={tuple(features.shape)}")
    if tuple(padding_mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"padding_mask must align with features [B,T], got {tuple(padding_mask.shape)} vs {tuple(features.shape[:2])}")
    valid = ~padding_mask.to(device=features.device, dtype=torch.bool)
    valid_count = valid.sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError("mask_aware_mean cannot pool samples with zero valid tokens")
    masked = features.masked_fill(~valid.unsqueeze(-1), 0.0)
    embedding = masked.sum(dim=1) / valid_count.unsqueeze(-1).to(dtype=features.dtype)
    return embedding.float(), valid_count


def pooled_logits_from_features(model: BEATs, features: Tensor, padding_mask: Tensor) -> Tensor:
    features = features.float()
    token_features = model.predictor_dropout(features) if hasattr(model, "predictor_dropout") else features
    token_logits = model.predictor(token_features.float()).float()
    if token_logits.ndim != 3:
        raise ValueError(f"predictor output must be [B,T,C], got shape={tuple(token_logits.shape)}")
    if tuple(token_logits.shape[:2]) != tuple(padding_mask.shape):
        raise ValueError("token logits and padding_mask [B,T] do not align")
    valid = ~padding_mask.to(device=token_logits.device, dtype=torch.bool)
    masked = token_logits.masked_fill(~valid.unsqueeze(-1), 0.0)
    denom = valid.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype=token_logits.dtype)
    return (masked.sum(dim=1) / denom).float()



@dataclass(frozen=True)
class _OerActivationBatch:
    hook_name: str
    layer_idx: int
    module_name: str
    activation_position: str
    tensor: Tensor
    valid_token_mask: Tensor
    original_layout: str
    original_shape: tuple[int, ...]
    canonical_layout: str
    canonical_shape: tuple[int, ...]
    layout_inference_source: str
    source: str


def _expected_oer_hook_list() -> list[dict[str, Any]]:
    return [
        {
            "hook_name": f"encoder.layers.{layer_idx}.{module_name}.{OER_ACTIVATION_POSITION_USED}",
            "module_path": f"encoder.layers.{layer_idx}.{module_name}",
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "activation_position": OER_ACTIVATION_POSITION_USED,
        }
        for layer_idx in range(OER_EXPECTED_NUM_LAYERS)
        for module_name in OER_FFN_MODULE_NAMES
    ]


def _oer_hook_list_hash(hook_list: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_hash({"oer_hook_list_used": [dict(item) for item in hook_list]})


def _baseline_oer_hook_list(model: nn.Module) -> list[dict[str, Any]]:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("Baseline OER requires model.encoder.layers")
    if int(len(layers)) != OER_EXPECTED_NUM_LAYERS:
        raise RuntimeError(
            f"Baseline full-FFN OER requires exactly {OER_EXPECTED_NUM_LAYERS} Transformer layers; got {int(len(layers))}"
        )
    hooks = _expected_oer_hook_list()
    missing: list[str] = []
    for hook in hooks:
        layer = layers[int(hook["layer_idx"])]
        if not hasattr(layer, str(hook["module_name"])):
            missing.append(str(hook["module_path"]))
    if missing:
        raise RuntimeError(f"Baseline OER missing FFN target module(s): {missing[:8]}")
    return hooks


def _get_oer_module(model: nn.Module, layer_idx: int, module_name: str) -> nn.Module:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("Baseline OER requires model.encoder.layers")
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= int(len(layers)):
        raise IndexError(f"OER layer_idx={layer_idx} outside model.encoder.layers")
    layer = layers[layer_idx]
    if not hasattr(layer, str(module_name)):
        raise RuntimeError(f"Baseline OER target encoder.layers.{layer_idx}.{module_name} missing")
    return getattr(layer, str(module_name))


def _canonicalize_oer_activation_btc(
    activation: Tensor,
    valid_token_mask: Tensor,
    *,
    hook_name: str,
    source: str,
) -> tuple[Tensor, Tensor, str, tuple[int, ...], str]:
    x = activation.detach().float()
    valid = valid_token_mask.detach().bool()
    if x.ndim != 3:
        raise ValueError(f"{hook_name}: OER activation must be 3D, got shape={tuple(x.shape)}")
    if valid.ndim != 2:
        raise ValueError(f"{hook_name}: OER valid_token_mask must be [B,T], got shape={tuple(valid.shape)}")
    bsz, num_tokens = int(valid.shape[0]), int(valid.shape[1])
    original_shape = tuple(int(item) for item in x.shape)
    if tuple(x.shape[:2]) == (bsz, num_tokens):
        btc = x
        original_layout = "BTC"
    elif tuple(x.shape[:2]) == (num_tokens, bsz):
        btc = x.transpose(0, 1)
        original_layout = "TBC"
    else:
        raise ValueError(
            f"{hook_name}: cannot align OER activation shape={tuple(x.shape)} with valid_token_mask [B,T]={tuple(valid.shape)}"
        )
    btc = btc.detach().cpu().float().contiguous()
    valid = valid.detach().cpu().bool().contiguous()
    _assert_finite_tensor(f"{hook_name}.{source}", btc, valid_token_mask=valid)
    return btc, valid, original_layout, original_shape, "shape_and_token_mask"


class _OerInputHookManager:
    def __init__(self, model: nn.Module, hook_list: Sequence[Mapping[str, Any]], token_padding_mask: Tensor) -> None:
        self.model = model
        self.hook_list = [dict(item) for item in hook_list]
        self.valid_token_mask = (~token_padding_mask.detach().bool()).cpu().contiguous()
        self._handles: list[Any] = []
        self._cache: dict[str, _OerActivationBatch] = {}

    def register_hooks(self) -> None:
        if self._handles:
            raise RuntimeError("Baseline OER hooks already registered")
        for hook in self.hook_list:
            module = _get_oer_module(self.model, int(hook["layer_idx"]), str(hook["module_name"]))
            hook_name = str(hook["hook_name"])
            layer_idx = int(hook["layer_idx"])
            module_name = str(hook["module_name"])

            def pre_hook(_module: nn.Module, inputs: tuple[Any, ...], *, _hook_name: str = hook_name, _layer_idx: int = layer_idx, _module_name: str = module_name) -> None:
                del _module
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise RuntimeError(f"{_hook_name}: OER forward_pre_hook did not receive a Tensor input")
                tensor, valid, original_layout, original_shape, inference_source = _canonicalize_oer_activation_btc(
                    inputs[0], self.valid_token_mask, hook_name=_hook_name, source="fp32_forward_pre_hook_input"
                )
                self._cache[_hook_name] = _OerActivationBatch(
                    hook_name=_hook_name,
                    layer_idx=int(_layer_idx),
                    module_name=str(_module_name),
                    activation_position=OER_ACTIVATION_POSITION_USED,
                    tensor=tensor,
                    valid_token_mask=valid,
                    original_layout=original_layout,
                    original_shape=original_shape,
                    canonical_layout="BTC",
                    canonical_shape=tuple(int(item) for item in tensor.shape),
                    layout_inference_source=inference_source,
                    source="fp32_forward_pre_hook_input",
                )

            self._handles.append(module.register_forward_pre_hook(pre_hook))

    def pop_cache(self) -> dict[str, _OerActivationBatch]:
        missing = [str(item["hook_name"]) for item in self.hook_list if str(item["hook_name"]) not in self._cache]
        if missing:
            raise RuntimeError(f"Baseline FP32 OER did not capture hook(s): {missing[:8]}")
        result = dict(self._cache)
        self._cache.clear()
        return result

    def clear_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache.clear()


def _clear_quant_oer_state(model: nn.Module, hook_list: Sequence[Mapping[str, Any]]) -> None:
    for hook in hook_list:
        module = _get_oer_module(model, int(hook["layer_idx"]), str(hook["module_name"]))
        clear = getattr(module, "clear_oer_activation_tensors", None)
        if callable(clear):
            clear()


def _collect_quant_oer_activations(model: nn.Module, hook_list: Sequence[Mapping[str, Any]], token_padding_mask: Tensor) -> dict[str, _OerActivationBatch]:
    expected_valid = (~token_padding_mask.detach().bool()).cpu().contiguous()
    result: dict[str, _OerActivationBatch] = {}
    for hook in hook_list:
        hook_name = str(hook["hook_name"])
        module = _get_oer_module(model, int(hook["layer_idx"]), str(hook["module_name"]))
        getter = getattr(module, "get_oer_activation_tensors", None)
        if not callable(getter):
            raise RuntimeError(f"{hook_name}: INT4/KD OER requires QuantLinear get_oer_activation_tensors()")
        _fp32_activation, current_activation = getter()
        if current_activation is None:
            raise RuntimeError(f"{hook_name}: QuantLinear did not expose current quant activation for OER")
        valid_mask = getattr(module, "last_activation_valid_mask_for_oer", None)
        if valid_mask is None:
            raise RuntimeError(f"{hook_name}: QuantLinear did not expose token-aligned valid mask for OER")
        tensor, valid, original_layout, original_shape, inference_source = _canonicalize_oer_activation_btc(
            current_activation, valid_mask, hook_name=hook_name, source="quantlinear_current_input_activation"
        )
        if tuple(valid.shape) != tuple(expected_valid.shape) or not bool(torch.equal(valid, expected_valid)):
            raise RuntimeError(f"{hook_name}: OER valid token mask differs from export token padding mask")
        result[hook_name] = _OerActivationBatch(
            hook_name=hook_name,
            layer_idx=int(hook["layer_idx"]),
            module_name=str(hook["module_name"]),
            activation_position=OER_ACTIVATION_POSITION_USED,
            tensor=tensor,
            valid_token_mask=valid,
            original_layout=original_layout,
            original_shape=original_shape,
            canonical_layout="BTC",
            canonical_shape=tuple(int(item) for item in tensor.shape),
            layout_inference_source=inference_source,
            source="quantlinear_current_input_activation",
        )
    _clear_quant_oer_state(model, hook_list)
    return result


def _oer_oracle_root(output_dirs: Mapping[str, Path], *, split: str, flat_output: bool) -> Path:
    root = Path(output_dirs["features"])
    return root / "oer_oracle" if flat_output else root / f"{split}_oer_oracle"


def _oer_oracle_paths_from_root(root: Path) -> tuple[Path, Path]:
    return root / OER_ORACLE_FILENAME, root / OER_ORACLE_MANIFEST_FILENAME


def _reference_oer_oracle_paths(reference_root: str | Path, *, split: str) -> tuple[Path, Path]:
    root = Path(reference_root)
    nonflat_root = root / "features" / REFERENCE_MODEL_NAME / f"{split}_oer_oracle"
    nonflat_paths = _oer_oracle_paths_from_root(nonflat_root)
    if nonflat_paths[0].exists() or nonflat_paths[1].exists():
        return nonflat_paths
    flat_root = root / "oer_oracle"
    flat_paths = _oer_oracle_paths_from_root(flat_root)
    if flat_paths[0].exists() or flat_paths[1].exists():
        return flat_paths
    feature_flat_root = root / "features" / "oer_oracle"
    return _oer_oracle_paths_from_root(feature_flat_root)


def _oer_topk_row(*, sample_id: str, activation: Tensor, valid_tokens: Tensor, hook: Mapping[str, Any]) -> dict[str, Any]:
    if activation.ndim != 2:
        raise ValueError(f"{hook['hook_name']}: per-sample OER activation must be [T,C], got {tuple(activation.shape)}")
    if valid_tokens.ndim != 1 or int(valid_tokens.numel()) != int(activation.shape[0]):
        raise ValueError(f"{hook['hook_name']}: valid token mask must align with activation tokens")
    feature_dim = int(activation.shape[1])
    flat = activation.detach().cpu().float().reshape(-1)
    valid_scalar_mask = valid_tokens.detach().cpu().bool().unsqueeze(-1).expand(int(activation.shape[0]), feature_dim).reshape(-1)
    valid_indices = torch.nonzero(valid_scalar_mask, as_tuple=False).reshape(-1).long()
    valid_scalar_count = int(valid_indices.numel())
    invalid_reason = ""
    if valid_scalar_count <= 0:
        topk_indices = torch.empty(0, dtype=torch.long)
        fp32_energy = float("nan")
        topk_k = 0
        invalid_reason = "no_valid_oracle_positions"
    else:
        topk_k = max(1, int(math.ceil(float(valid_scalar_count) * float(OER_TOPK_RATIO))))
        values = flat[valid_indices].square()
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError(f"{hook['hook_name']}: non-finite FP32 OER activation energy for sample_id={sample_id}")
        local_indices = torch.topk(values, k=topk_k, largest=True, sorted=False).indices
        topk_indices = valid_indices[local_indices].long().cpu().contiguous()
        fp32_energy_tensor = flat[topk_indices].square().sum()
        fp32_energy = float(fp32_energy_tensor.item())
        if not math.isfinite(fp32_energy):
            raise FloatingPointError(f"{hook['hook_name']}: non-finite FP32 oracle energy for sample_id={sample_id}")
        if fp32_energy <= float(OER_EPS):
            invalid_reason = "fp32_oracle_energy_le_eps"
    return {
        "sample_id": str(sample_id),
        "hook_name": str(hook["hook_name"]),
        "layer_idx": int(hook["layer_idx"]),
        "module_name": str(hook["module_name"]),
        "activation_position": OER_ACTIVATION_POSITION_USED,
        "activation_layout": "BTC",
        "activation_shape": [int(activation.shape[0]), int(activation.shape[1])],
        "feature_dim": int(feature_dim),
        "valid_token_count": int(valid_tokens.detach().cpu().bool().sum().item()),
        "valid_scalar_count": int(valid_scalar_count),
        "topk_ratio": float(OER_TOPK_RATIO),
        "topk_k": int(topk_k),
        "topk_scalar_indices": topk_indices,
        "fp32_oracle_energy": float(fp32_energy),
        "invalid_reason": invalid_reason,
    }


class _OerOracleWriter:
    def __init__(
        self,
        *,
        root: Path,
        split: str,
        seed: int,
        model_name: str,
        run_identity_hash: str,
        run_manifest_path: Path,
        checkpoint_path: Path,
        checkpoint_sha256: str | None,
        label_mapping_hash: str,
        split_hash: str,
        hook_list: Sequence[Mapping[str, Any]],
    ) -> None:
        self.root = Path(root)
        self.shard_root = self.root / "shards"
        self.split = str(split)
        self.seed = int(seed)
        self.model_name = str(model_name)
        self.run_identity_hash = str(run_identity_hash)
        self.run_manifest_path = Path(run_manifest_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_sha256 = "" if checkpoint_sha256 is None else str(checkpoint_sha256)
        self.label_mapping_hash = str(label_mapping_hash)
        self.split_hash = str(split_hash)
        self.hook_list = [dict(item) for item in hook_list]
        self.hook_hash = _oer_hook_list_hash(self.hook_list)
        self.shards: list[dict[str, Any]] = []
        self.sample_index: list[dict[str, Any]] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_root.mkdir(parents=True, exist_ok=True)

    def write_shard(self, *, shard_id: int, start_row: int, end_row: int, sample_ids: Sequence[str], activations: Mapping[str, _OerActivationBatch]) -> None:
        rows: list[dict[str, Any]] = []
        for hook in self.hook_list:
            hook_name = str(hook["hook_name"])
            capture = activations.get(hook_name)
            if capture is None:
                raise RuntimeError(f"FP32 OER oracle missing activation capture for {hook_name}")
            if int(capture.tensor.shape[0]) != len(sample_ids):
                raise RuntimeError(f"{hook_name}: OER activation batch size does not match sample ids")
            for sample_offset, sample_id in enumerate(sample_ids):
                rows.append(_oer_topk_row(sample_id=str(sample_id), activation=capture.tensor[int(sample_offset)], valid_tokens=capture.valid_token_mask[int(sample_offset)], hook=hook))
        shard_path = self.shard_root / f"shard_{int(shard_id):05d}.pt"
        payload = {
            "artifact_type": "baseline_fp32_oer_oracle_shard",
            "protocol_version": OER_PROTOCOL_VERSION,
            "model_kind": REFERENCE_MODEL_NAME,
            "split": self.split,
            "seed": int(self.seed),
            "run_identity_hash": self.run_identity_hash,
            "run_manifest_path": _path_str(self.run_manifest_path),
            "shard_id": int(shard_id),
            "start_row": int(start_row),
            "end_row": int(end_row),
            "sample_id": [str(item) for item in sample_ids],
            "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
            "oer_hook_list_used_hash": self.hook_hash,
            "rows": rows,
        }
        torch.save(payload, shard_path)
        shard_record = {
            "shard_id": int(shard_id),
            "start_row": int(start_row),
            "end_row": int(end_row),
            "sample_id_start": str(sample_ids[0]) if sample_ids else "",
            "sample_id_end": str(sample_ids[-1]) if sample_ids else "",
            "row_count": int(len(rows)),
            "file_path": _path_str(shard_path),
            "file_sha256": file_sha256(shard_path),
        }
        self.shards.append(shard_record)
        for sample_id in sample_ids:
            self.sample_index.append({"sample_id": str(sample_id), "shard_id": int(shard_id)})

    def finish(self, *, sample_ids: Sequence[str], sample_order_hash: str) -> dict[str, Any]:
        oracle_path, manifest_path = _oer_oracle_paths_from_root(self.root)
        manifest = {
            "protocol_version": OER_PROTOCOL_VERSION,
            "artifact_type": "baseline_fp32_oer_oracle_manifest",
            "model_kind": REFERENCE_MODEL_NAME,
            "model_name": REFERENCE_MODEL_NAME,
            "split": self.split,
            "seed": int(self.seed),
            "run_identity_hash": self.run_identity_hash,
            "run_manifest_path": _path_str(self.run_manifest_path),
            "oracle_source_checkpoint": _path_str(self.checkpoint_path),
            "oracle_source_checkpoint_sha256": self.checkpoint_sha256,
            "oracle_hook_list_hash": self.hook_hash,
            "oracle_granularity": "scalar",
            "oracle_energy_formula": "sum(H[t,c]^2) over FP32 top-k scalar positions among token-aligned valid tokens",
            "sample_order_hash": str(sample_order_hash),
            "label_mapping_hash": self.label_mapping_hash,
            "split_hash": self.split_hash,
            "padding_mask_protocol": "BEATs token-aligned padding_mask after frontend/patch downsampling; True means padding; OER excludes padding tokens",
            "activation_layout": "BTC",
            "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
            "oer_hook_list_used": self.hook_list,
            "oer_hook_list_used_hash": self.hook_hash,
            "topk_ratio": float(OER_TOPK_RATIO),
            "eps": float(OER_EPS),
            "num_samples": int(len(sample_ids)),
            "num_hooks": int(len(self.hook_list)),
            "expected_num_hooks": int(OER_EXPECTED_NUM_LAYERS * len(OER_FFN_MODULE_NAMES)),
            "sample_id_order": [str(item) for item in sample_ids],
            "sharded": True,
            "shards": list(self.shards),
        }
        payload = {
            "artifact_type": "baseline_fp32_oer_oracle",
            "protocol_version": OER_PROTOCOL_VERSION,
            "model_kind": REFERENCE_MODEL_NAME,
            "split": self.split,
            "seed": int(self.seed),
            "run_identity_hash": self.run_identity_hash,
            "manifest_path": _path_str(manifest_path),
            "manifest": manifest,
            "sample_index": list(self.sample_index),
            "shards": list(self.shards),
            "oer_hook_list_used": self.hook_list,
            "oer_hook_list_used_hash": self.hook_hash,
        }
        torch.save(payload, oracle_path)
        manifest = {**manifest, "oracle_path": _path_str(oracle_path), "oracle_artifact_sha256": file_sha256(oracle_path)}
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return {"path": _path_str(oracle_path), "manifest_path": _path_str(manifest_path), "sha256": file_sha256(oracle_path), "manifest_sha256": file_sha256(manifest_path), "manifest": manifest}


def _required_oer_manifest_field(manifest: Mapping[str, Any], field: str, *, manifest_path: Path) -> Any:
    value = manifest.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"OER oracle manifest missing required field {field!r}: {manifest_path}")
    return value


def _cross_platform_path_name(path_value: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if "\\" in raw or (len(raw) >= 2 and raw[1] == ":"):
        return PureWindowsPath(raw).name
    return Path(raw).name


def _portable_path_parts(path_value: str | Path) -> tuple[str, ...]:
    raw = str(path_value or "").strip()
    if not raw:
        return ()
    return tuple(part for part in raw.replace("\\", "/").split("/") if part and part != ".")


def _looks_like_windows_absolute_path(path_value: str) -> bool:
    raw = str(path_value or "").strip()
    return len(raw) >= 3 and raw[1] == ":" and raw[2] in ("\\", "/")


def _oer_oracle_shard_path_candidates(*, recorded_path: str, oracle_path: Path, manifest_path: Path) -> list[Path]:
    raw = str(recorded_path or "").strip()
    if not raw:
        return []
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)

    direct = Path(raw)
    add(direct)
    if not direct.is_absolute() and not _looks_like_windows_absolute_path(raw):
        add(oracle_path.parent / direct)
        add(manifest_path.parent / direct)

    filename = _cross_platform_path_name(raw)
    if filename:
        add(oracle_path.parent / "shards" / filename)
        add(manifest_path.parent / "shards" / filename)
        add(oracle_path.parent / filename)
        add(manifest_path.parent / filename)
    return candidates


def _resolve_oer_oracle_shard_path(
    *,
    recorded_path: str,
    oracle_path: Path,
    manifest_path: Path,
    expected_sha256: str,
    shard_id: int,
) -> tuple[Path, str | None, bool]:
    candidates = _oer_oracle_shard_path_candidates(
        recorded_path=recorded_path,
        oracle_path=oracle_path,
        manifest_path=manifest_path,
    )
    hash_mismatches: list[str] = []
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        actual_sha256 = file_sha256(candidate)
        if expected_sha256 and actual_sha256 != expected_sha256:
            hash_mismatches.append(f"{candidate} actual={actual_sha256}")
            continue
        resolved = candidate.resolve(strict=False)
        migrated = str(resolved) != str(Path(recorded_path).resolve(strict=False))
        return resolved, actual_sha256, bool(migrated)

    checked = ", ".join(str(item) for item in candidates[:8])
    if hash_mismatches:
        details = "; ".join(hash_mismatches[:4])
        raise ValueError(
            f"OER oracle shard hash mismatch for shard_id={int(shard_id)}: "
            f"expected={expected_sha256}; checked={details}"
        )
    raise FileNotFoundError(
        f"OER oracle shard missing for shard_id={int(shard_id)}. "
        f"recorded_path={recorded_path}; checked={checked}"
    )


def _nanmean_float_values(values: Sequence[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), 0
    return float(sum(finite) / len(finite)), int(len(finite))


def _current_oer_ratio_for_row(*, oracle_row: Mapping[str, Any], current_activation: Tensor, current_valid_tokens: Tensor, sample_id: str, hook_name: str) -> tuple[float, str]:
    if str(oracle_row.get("sample_id")) != str(sample_id):
        raise ValueError(f"OER oracle sample_id mismatch for hook={hook_name}")
    if str(oracle_row.get("hook_name")) != str(hook_name):
        raise ValueError(f"OER oracle hook_name mismatch for sample_id={sample_id}")
    if str(oracle_row.get("activation_position")) != OER_ACTIVATION_POSITION_USED:
        raise ValueError(f"OER oracle activation position mismatch for sample_id={sample_id} hook={hook_name}")
    if str(oracle_row.get("activation_layout")) != "BTC":
        raise ValueError(f"OER oracle activation layout mismatch for sample_id={sample_id} hook={hook_name}")
    expected_shape = [int(item) for item in oracle_row.get("activation_shape", [])]
    current_shape = [int(current_activation.shape[0]), int(current_activation.shape[1])]
    if expected_shape != current_shape:
        raise ValueError(f"OER activation shape mismatch for sample_id={sample_id} hook={hook_name}: oracle={expected_shape} current={current_shape}")
    topk = torch.as_tensor(oracle_row.get("topk_scalar_indices"), dtype=torch.long).reshape(-1)
    denom = float(oracle_row.get("fp32_oracle_energy", float("nan")))
    if int(topk.numel()) <= 0:
        return float("nan"), str(oracle_row.get("invalid_reason") or "no_valid_oracle_positions")
    if not math.isfinite(denom) or denom <= float(OER_EPS):
        return float("nan"), str(oracle_row.get("invalid_reason") or "fp32_oracle_energy_le_eps")
    feature_dim = int(oracle_row.get("feature_dim", current_activation.shape[1]))
    if feature_dim != int(current_activation.shape[1]):
        raise ValueError(f"OER feature_dim mismatch for sample_id={sample_id} hook={hook_name}")
    max_index = int(current_activation.numel()) - 1
    if int(topk.min().item()) < 0 or int(topk.max().item()) > max_index:
        raise ValueError(f"OER oracle topk index out of current activation range for sample_id={sample_id} hook={hook_name}")
    token_indices = torch.div(topk, feature_dim, rounding_mode="floor").long()
    valid = current_valid_tokens.detach().cpu().bool().reshape(-1)
    if int(valid.numel()) != int(current_activation.shape[0]):
        raise ValueError(f"OER current valid mask token length mismatch for sample_id={sample_id} hook={hook_name}")
    if not bool(valid[token_indices].all().item()):
        raise ValueError(f"OER oracle top-k includes padding token under current mask for sample_id={sample_id} hook={hook_name}")
    flat = current_activation.detach().cpu().float().reshape(-1)
    numerator = float(flat[topk].square().sum().item())
    if not math.isfinite(numerator):
        raise FloatingPointError(f"non-finite current OER energy for sample_id={sample_id} hook={hook_name}")
    return float(numerator / denom), ""


class _OerOracleReader:
    def __init__(
        self,
        *,
        oracle_path: Path,
        manifest_path: Path,
        split: str,
        hook_list: Sequence[Mapping[str, Any]],
        label_mapping_hash: str,
        split_hash: str,
        expected_checkpoint_hash: str | None,
    ) -> None:
        self.oracle_path = Path(oracle_path)
        self.manifest_path = Path(manifest_path)
        if not self.oracle_path.exists() or not self.oracle_path.is_file():
            raise FileNotFoundError(f"FP32 OER oracle artifact missing: {self.oracle_path}")
        if not self.manifest_path.exists() or not self.manifest_path.is_file():
            raise FileNotFoundError(f"FP32 OER oracle manifest missing: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, Mapping):
            raise TypeError(f"OER oracle manifest must be a mapping: {self.manifest_path}")
        for field in (
            "protocol_version",
            "oracle_source_checkpoint_sha256",
            "oracle_hook_list_hash",
            "oracle_granularity",
            "oracle_energy_formula",
            "sample_order_hash",
            "label_mapping_hash",
            "split_hash",
            "padding_mask_protocol",
            "activation_layout",
            "oer_activation_position_used",
            "oer_hook_list_used",
        ):
            _required_oer_manifest_field(manifest, field, manifest_path=self.manifest_path)
        if str(manifest["protocol_version"]) != OER_PROTOCOL_VERSION:
            raise ValueError(f"OER oracle protocol mismatch: {manifest['protocol_version']!r} != {OER_PROTOCOL_VERSION!r}")
        if str(manifest.get("model_kind")) != REFERENCE_MODEL_NAME:
            raise ValueError("OER oracle must be produced by Baseline FP32 teacher export")
        if str(manifest.get("split")) != str(split):
            raise ValueError(f"OER oracle split mismatch: {manifest.get('split')!r} != {split!r}")
        if str(manifest.get("label_mapping_hash")) != str(label_mapping_hash):
            raise ValueError("OER oracle label_mapping_hash differs from current export")
        if str(manifest.get("split_hash")) != str(split_hash):
            raise ValueError("OER oracle split_hash differs from current export")
        if str(manifest.get("activation_layout")) != "BTC":
            raise ValueError("OER oracle activation_layout must be BTC")
        if str(manifest.get("oer_activation_position_used")) != OER_ACTIVATION_POSITION_USED:
            raise ValueError("OER oracle activation position must be FFN input")
        expected_hook_hash = _oer_hook_list_hash(hook_list)
        if str(manifest.get("oracle_hook_list_hash")) != expected_hook_hash or str(manifest.get("oer_hook_list_used_hash")) != expected_hook_hash:
            raise ValueError("OER oracle hook list hash differs from Baseline canonical FFN input hook list")
        expected_hash = "" if expected_checkpoint_hash is None else str(expected_checkpoint_hash).strip()
        if expected_hash and str(manifest.get("oracle_source_checkpoint_sha256")) != expected_hash:
            raise ValueError("OER oracle source checkpoint hash differs from expected FP32 teacher checkpoint")
        actual_oracle_hash = file_sha256(self.oracle_path)
        declared_oracle_hash = str(manifest.get("oracle_artifact_sha256") or "")
        if declared_oracle_hash and actual_oracle_hash != declared_oracle_hash:
            raise ValueError(f"OER oracle artifact hash mismatch: manifest={declared_oracle_hash} actual={actual_oracle_hash}")
        oracle = torch.load(self.oracle_path, map_location="cpu")
        if not isinstance(oracle, Mapping):
            raise TypeError(f"OER oracle artifact must be a mapping: {self.oracle_path}")
        self.manifest = dict(manifest)
        self.oracle = dict(oracle)
        sample_index = oracle.get("sample_index")
        if not isinstance(sample_index, Sequence) or isinstance(sample_index, (str, bytes)):
            raise ValueError("OER oracle artifact missing sample_index")
        self.sample_to_shard: dict[str, int] = {}
        for item in sample_index:
            if not isinstance(item, Mapping):
                raise ValueError("OER oracle sample_index entries must be mappings")
            sample_id = str(item.get("sample_id", ""))
            if not sample_id:
                raise ValueError("OER oracle sample_index entry missing sample_id")
            if sample_id in self.sample_to_shard:
                raise ValueError(f"OER oracle duplicate sample_id in sample_index: {sample_id}")
            self.sample_to_shard[sample_id] = int(item.get("shard_id"))
        self.shard_by_id = {int(item["shard_id"]): dict(item) for item in oracle.get("shards", []) if isinstance(item, Mapping)}
        if not self.shard_by_id:
            raise ValueError("OER oracle artifact missing shard index")
        self._shard_cache: dict[int, Mapping[str, Any]] = {}
        self.shard_path_migrations: dict[int, dict[str, str]] = {}
        self.hook_list = [dict(item) for item in hook_list]

    def _load_shard(self, shard_id: int) -> Mapping[str, Any]:
        shard_id = int(shard_id)
        cached = self._shard_cache.get(shard_id)
        if cached is not None:
            return cached
        shard_info = self.shard_by_id.get(shard_id)
        if shard_info is None:
            raise ValueError(f"OER oracle missing shard_id={shard_id}")
        expected_hash = str(shard_info.get("file_sha256") or "")
        recorded_path = str(shard_info.get("file_path") or "").strip()
        if not recorded_path:
            raise ValueError(f"OER oracle shard_id={shard_id} missing file_path")
        shard_path, actual_hash, migrated = _resolve_oer_oracle_shard_path(
            recorded_path=recorded_path,
            oracle_path=self.oracle_path,
            manifest_path=self.manifest_path,
            expected_sha256=expected_hash,
            shard_id=shard_id,
        )
        if migrated:
            self.shard_path_migrations[shard_id] = {
                "recorded_path": recorded_path,
                "resolved_path": str(shard_path),
            }
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"OER oracle shard hash mismatch for {shard_path}: manifest={expected_hash} actual={actual_hash}")
        payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"OER oracle shard must be a mapping: {shard_path}")
        self._shard_cache[shard_id] = payload
        return payload

    def validate_final_sample_order(self, sample_ids: Sequence[str], sample_order_hash: str) -> None:
        if str(self.manifest.get("sample_order_hash")) != str(sample_order_hash):
            raise ValueError("OER oracle sample_order_hash differs from current export")
        oracle_order = [str(item) for item in self.manifest.get("sample_id_order", [])]
        if oracle_order and oracle_order != [str(item) for item in sample_ids]:
            mismatch = [idx for idx, (cur, ref) in enumerate(zip(sample_ids, oracle_order)) if str(cur) != str(ref)][:8]
            raise ValueError(f"OER oracle sample_id_order differs from current export; mismatch indices={mismatch}")

    def records_for_batch(self, *, sample_ids: Sequence[str], labels: Tensor, buckets: Sequence[str], activations: Mapping[str, _OerActivationBatch]) -> list[dict[str, Any]]:
        labels_cpu = labels.detach().cpu().long().reshape(-1)
        if len(sample_ids) != int(labels_cpu.numel()) or len(sample_ids) != len(buckets):
            raise ValueError("OER record batch metadata dimensions do not align")
        needed_shards = {self.sample_to_shard.get(str(sample_id)) for sample_id in sample_ids}
        if None in needed_shards:
            missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in self.sample_to_shard]
            raise ValueError(f"OER oracle missing sample_id(s): {missing[:8]}")
        row_map: dict[tuple[str, str], Mapping[str, Any]] = {}
        for shard_id in sorted(int(item) for item in needed_shards if item is not None):
            shard = self._load_shard(shard_id)
            for row in shard.get("rows", []):
                if isinstance(row, Mapping):
                    row_map[(str(row.get("sample_id")), str(row.get("hook_name")))] = row
        records: list[dict[str, Any]] = []
        expected_hook_count = int(len(self.hook_list))
        for sample_offset, sample_id in enumerate(sample_ids):
            ratios: list[float] = []
            invalid_reasons: dict[str, int] = {}
            for hook in self.hook_list:
                hook_name = str(hook["hook_name"])
                oracle_row = row_map.get((str(sample_id), hook_name))
                if oracle_row is None:
                    raise ValueError(f"OER oracle missing row for sample_id={sample_id} hook={hook_name}")
                capture = activations.get(hook_name)
                if capture is None:
                    raise RuntimeError(f"current export missing OER activation for hook={hook_name}")
                ratio, reason = _current_oer_ratio_for_row(
                    oracle_row=oracle_row,
                    current_activation=capture.tensor[int(sample_offset)],
                    current_valid_tokens=capture.valid_token_mask[int(sample_offset)],
                    sample_id=str(sample_id),
                    hook_name=hook_name,
                )
                ratios.append(float(ratio))
                if reason:
                    invalid_reasons[reason] = int(invalid_reasons.get(reason, 0)) + 1
            value, valid_count = _nanmean_float_values(ratios)
            abs_deviation = abs(value - 1.0) if math.isfinite(value) else float("nan")
            records.append(
                {
                    "sample_id": str(sample_id),
                    "label_id": int(labels_cpu[int(sample_offset)].item()),
                    "bucket": str(buckets[int(sample_offset)]),
                    "oer_retention": float(value),
                    "oer_abs_deviation_from_1": float(abs_deviation),
                    "oer_valid_rate": float(valid_count) / float(expected_hook_count) if expected_hook_count else float("nan"),
                    "valid_hook_count": int(valid_count),
                    "hook_count": int(expected_hook_count),
                    "invalid_hook_count": int(expected_hook_count - valid_count),
                    "invalid_reasons": json.dumps(invalid_reasons, sort_keys=True, separators=(",", ":")),
                }
            )
        return records

def _assert_finite_tensor(name: str, tensor: Tensor, *, valid_token_mask: Tensor | None = None) -> None:
    if valid_token_mask is None:
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return
    if tensor.ndim != 3:
        raise ValueError(f"{name} valid-token finite check expects [B,T,C], got {tuple(tensor.shape)}")
    mask = valid_token_mask.to(device=tensor.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(tensor.shape[:2]):
        raise ValueError(f"{name} valid_token_mask must align with [B,T], got {tuple(mask.shape)} vs {tuple(tensor.shape[:2])}")
    invalid = ~torch.isfinite(tensor)
    if bool((invalid & mask.unsqueeze(-1)).any().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf in valid token region")


def _assert_no_parameter_grads(model: nn.Module, *, context: str) -> None:
    offenders = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if offenders:
        raise RuntimeError(
            f"{context} requires param.grad is None for every parameter before formal export; "
            f"non-empty grads found for {len(offenders)} parameter(s), examples={offenders[:8]}"
        )


def _build_finetuned_beats(cfg_payload: Mapping[str, Any], label_mapping: Mapping[str, int]) -> BEATs:
    cfg = copy.deepcopy(dict(cfg_payload))
    num_classes = num_predictor_classes(label_mapping)
    cfg["finetuned_model"] = True
    cfg["predictor_class"] = int(num_classes)
    model = BEATs(BEATsConfig(cfg))
    if not hasattr(model, "predictor_dropout"):
        model.predictor_dropout = nn.Dropout(float(cfg.get("predictor_dropout", 0.1)))
    model.predictor = nn.Linear(int(model.cfg.encoder_embed_dim), int(num_classes))
    model.cfg.finetuned_model = True
    model.cfg.predictor_class = int(num_classes)
    return model


def _normalise_checkpoint_label_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise KeyError("checkpoint must contain a non-empty label_mapping")
    result = {str(label_name): int(label_id) for label_name, label_id in value.items()}
    _validate_dense_label_mapping(result, context="checkpoint label_mapping")
    return result


def _infer_checkpoint_model_kind(checkpoint: Mapping[str, Any]) -> str:
    raw = checkpoint.get("model_kind") or checkpoint.get("checkpoint_role") or checkpoint.get("artifact_type")
    text = "" if raw is None else str(raw).strip().lower()
    if text in {"fp32_teacher", "dataset_fp32_teacher"} or "fp32_teacher" in text:
        return "fp32_teacher"
    if text in {"int4_student_init", "int4_student"} or "int4" in text:
        return "int4_student_init"
    if text == "vanilla_kd" or "vanilla_kd" in text:
        return "vanilla_kd"
    if text == "two_stage_decoupling" or "two_stage_decoupling" in text:
        return "two_stage_decoupling"
    if text == "two_stage_class_prior_tau" or "two_stage_class_prior_tau" in text:
        return "two_stage_class_prior_tau"
    if text == "two_stage_tail_repair" or "two_stage_tail_repair" in text:
        return "two_stage_tail_repair"
    if "cfg" in checkpoint and "model" in checkpoint:
        return "fp32_teacher"
    if "cfg" in checkpoint and "model_state_dict" in checkpoint and "qparams" in checkpoint:
        return "vanilla_kd" if "kd_config" in checkpoint else "int4_student_init"
    raise KeyError(
        "cannot infer checkpoint model_kind; expected fp32_teacher/int4_student_init/"
        "vanilla_kd/two_stage_decoupling/two_stage_class_prior_tau/two_stage_tail_repair"
    )


def _require_checkpoint_keys(checkpoint: Mapping[str, Any], keys: Sequence[str], *, model_kind: str) -> None:
    missing = [key for key in keys if key not in checkpoint]
    if missing:
        raise KeyError(f"{model_kind} checkpoint missing required key(s): {missing}")


def _checkpoint_path_value(checkpoint_path: Path, value: Any) -> Path:
    target = Path(str(value))
    if target.is_absolute():
        return target
    return checkpoint_path.parent / target


def _verify_checkpoint_qparams(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    qparams_csv_override: Path | None = None,
) -> tuple[Mapping[str, Any], str, Path]:
    if "qparams" not in checkpoint or not isinstance(checkpoint.get("qparams"), Mapping) or not checkpoint.get("qparams"):
        raise KeyError("INT4/KD checkpoint must contain non-empty qparams for automatic reconstruction")
    if not checkpoint.get("qparams_hash"):
        raise KeyError("INT4/KD checkpoint must contain qparams_hash")
    qparams_payload = checkpoint["qparams"]
    computed_hash = qparams_hash(qparams_payload)
    stored_hash = str(checkpoint.get("qparams_hash"))
    if computed_hash != stored_hash:
        raise RuntimeError(f"checkpoint qparams_hash mismatch: stored={stored_hash}, computed={computed_hash}")
    if qparams_csv_override is None and not checkpoint.get("qparams_csv"):
        raise KeyError("INT4/KD checkpoint must contain qparams_csv")
    qparams_csv = (
        Path(qparams_csv_override)
        if qparams_csv_override is not None
        else _checkpoint_path_value(checkpoint_path, checkpoint["qparams_csv"])
    )
    if not qparams_csv.exists():
        raise FileNotFoundError(f"checkpoint qparams_csv does not exist: {qparams_csv}")
    csv_qparams = load_qparams_csv(qparams_csv)
    csv_qparams_hash = qparams_hash(csv_qparams)
    if csv_qparams_hash != computed_hash:
        raise RuntimeError(
            f"checkpoint qparams and qparams_csv disagree: checkpoint={computed_hash}, csv={csv_qparams_hash}"
        )
    stored_csv_hash = checkpoint.get("qparams_csv_sha256") or checkpoint.get("qparams_csv_hash")
    actual_csv_file_hash = file_sha256(qparams_csv)
    if stored_csv_hash and actual_csv_file_hash and str(stored_csv_hash) != str(actual_csv_file_hash):
        raise RuntimeError(
            f"checkpoint qparams_csv sha256 mismatch: stored={stored_csv_hash}, actual={actual_csv_file_hash}"
        )
    return qparams_payload, computed_hash, qparams_csv


def _checkpoint_quantized_layers(checkpoint: Mapping[str, Any], *, model_kind: str) -> list[dict[str, Any]]:
    raw = checkpoint.get("quantized_layers")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise KeyError(f"{model_kind} checkpoint must contain non-empty quantized_layers")
    layers = [dict(item) for item in raw]
    for index, row in enumerate(layers):
        if not row.get("module_path"):
            raise KeyError(f"quantized_layers[{index}] missing module_path")
    return layers


def _apply_checkpoint_qparams(controller: Any, qparams: Mapping[str, Any]) -> None:
    missing = sorted(set(controller.modules) - set(qparams))
    extra = sorted(set(qparams) - set(controller.modules))
    if missing or extra:
        raise RuntimeError(f"checkpoint qparams target mismatch: missing={missing}, extra={extra}")
    for module_path, module in sorted(controller.modules.items()):
        module_qparams = qparams[module_path]
        if not isinstance(module_qparams, Mapping):
            raise TypeError(f"qparams[{module_path!r}] must be a mapping")
        apply_fixed_qparams_to_quant_module(
            module,
            mode="w4a4",
            weight_qparams=module_qparams.get("weight"),
            activation_qparams=module_qparams.get("activation"),
            require_qparams=True,
        )
    controller.qparams_frozen = True
    controller.set_observer_enabled(False)
    controller.set_fake_quant_enabled(True)


def _module_quant_summary(name: str, module: nn.Module) -> dict[str, Any]:
    observer = getattr(module, "activation_observer", None)
    weight_scale = getattr(module, "weight_scale", None)
    activation_scale = getattr(module, "activation_scale", None)
    row = {
        "module_path": str(name),
        "observer_enabled": bool(getattr(observer, "enabled", False)),
        "calibration_enabled": bool(getattr(module, "calibration_enabled", False)),
        "fake_quant_enabled": bool(getattr(module, "fake_quant_enabled", False)),
        "qparams_frozen": bool(getattr(module, "qparams_frozen", False)),
        "weight_quant_enabled": bool(getattr(module, "weight_quant_enabled", False)),
        "activation_quant_enabled": bool(getattr(module, "activation_quant_enabled", False)),
        "weight_qparams_finite": None,
        "activation_qparams_finite": None,
    }
    if isinstance(weight_scale, Tensor) and int(weight_scale.numel()) > 0:
        row["weight_qparams_finite"] = bool(torch.isfinite(weight_scale.detach()).all().item()) and bool((weight_scale.detach() > 0).all().item())
    if isinstance(activation_scale, Tensor) and int(activation_scale.numel()) > 0:
        row["activation_qparams_finite"] = bool(torch.isfinite(activation_scale.detach()).all().item()) and bool((activation_scale.detach() > 0).all().item())
    return row


def audit_quant_state(
    model: nn.Module,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    quant_controller: Any | None = None,
    expected_quantized_layers: Sequence[Mapping[str, Any]] | None = None,
    expected_qparams_hash: str | None = None,
) -> dict[str, Any]:
    if quant_controller is None:
        quant_controller = getattr(model, "_baseline_quant_controller", None)
    controller_payload: Any = None
    audit_source = "module_attributes"
    if quant_controller is not None:
        audit_source = "quant_controller"
        controller_audit = getattr(quant_controller, "audit", None)
        if callable(controller_audit):
            controller_payload = controller_audit(model)
        elif hasattr(quant_controller, "state_summary"):
            controller_payload = quant_controller.state_summary()

    module_rows = [_module_quant_summary(name, module) for name, module in iter_quantized_linears(model)]
    module_names = sorted(row["module_path"] for row in module_rows)
    expected_layers = [dict(item) for item in (expected_quantized_layers or [])]
    expected_names = sorted(str(row.get("module_path", "")) for row in expected_layers if row.get("module_path"))
    if not expected_names and checkpoint is not None and checkpoint.get("quantized_layers"):
        expected_layers = [dict(item) for item in checkpoint["quantized_layers"]]
        expected_names = sorted(str(row.get("module_path", "")) for row in expected_layers if row.get("module_path"))

    if not module_rows and not expected_names:
        return {
            "quant_audit_source": "not_quantized",
            "num_quantized_modules": 0,
            "observer_enabled": None,
            "fake_quant_enabled": None,
            "qparams_frozen": None,
            "quantized_layers_match_checkpoint": True,
            "qparams_hash_match_checkpoint": True,
        }
    if module_names != expected_names:
        raise RuntimeError(f"quantized layer mismatch: model={module_names}, checkpoint={expected_names}")

    bad_observer = [row for row in module_rows if row["observer_enabled"] or row["calibration_enabled"]]
    bad_fake = [row for row in module_rows if not row["fake_quant_enabled"]]
    bad_frozen = [row for row in module_rows if not row["qparams_frozen"]]
    bad_weight_qparams = [row for row in module_rows if row["weight_quant_enabled"] and row["weight_qparams_finite"] is not True]
    bad_activation_qparams = [row for row in module_rows if row["activation_quant_enabled"] and row["activation_qparams_finite"] is not True]
    if bad_observer:
        raise RuntimeError(f"quant audit failed: observer/calibration still enabled for {bad_observer[:4]}")
    if bad_fake:
        raise RuntimeError(f"quant audit failed: fake_quant_enabled is false for {bad_fake[:4]}")
    if bad_frozen:
        raise RuntimeError(f"quant audit failed: qparams_frozen is false for {bad_frozen[:4]}")
    if bad_weight_qparams or bad_activation_qparams:
        raise RuntimeError(
            f"quant audit failed: non-finite or missing qparams; "
            f"weight={bad_weight_qparams[:4]}, activation={bad_activation_qparams[:4]}"
        )

    if isinstance(controller_payload, Mapping):
        if bool(controller_payload.get("observer_enabled", False)):
            raise RuntimeError("quant audit failed: controller observer_enabled is true")
        if not bool(controller_payload.get("fake_quant_enabled", True)):
            raise RuntimeError("quant audit failed: controller fake_quant_enabled is false")
        if not bool(controller_payload.get("qparams_frozen", True)):
            raise RuntimeError("quant audit failed: controller qparams_frozen is false")

    qparams_hash_match = True
    checkpoint_qparams_hash = None
    if checkpoint is not None and checkpoint.get("qparams_hash"):
        checkpoint_qparams_hash = str(checkpoint.get("qparams_hash"))
        if expected_qparams_hash is not None and checkpoint_qparams_hash != str(expected_qparams_hash):
            qparams_hash_match = False
            raise RuntimeError(
                f"quant audit failed: expected_qparams_hash={expected_qparams_hash} differs from checkpoint={checkpoint_qparams_hash}"
            )
    live_qparams_hash = None
    if quant_controller is not None and expected_qparams_hash is not None:
        live_qparams_hash = qparams_hash(collect_qparams_from_controller(quant_controller))
        if live_qparams_hash != str(expected_qparams_hash):
            qparams_hash_match = False
            raise RuntimeError(
                f"quant audit failed: live qparams hash={live_qparams_hash} differs from expected={expected_qparams_hash}"
            )

    return {
        "quant_audit_source": audit_source,
        "controller_audit": controller_payload,
        "num_quantized_modules": int(len(module_rows)),
        "target_names": module_names,
        "observer_enabled": False,
        "fake_quant_enabled": True,
        "qparams_frozen": True,
        "quantized_layers_match_checkpoint": True,
        "qparams_hash_match_checkpoint": bool(qparams_hash_match),
        "live_qparams_hash": live_qparams_hash,
        "checkpoint_qparams_hash": checkpoint_qparams_hash,
        "module_summaries": module_rows,
    }

def forward_export_batch(
    model: BEATs,
    batch: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
    token_layer_indices: Sequence[int],
    device: torch.device,
    oer_mode: str = "none",
    oer_hook_list: Sequence[Mapping[str, Any]] = (),
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Any], dict[str, _OerActivationBatch]]:
    moved = move_batch_to_device(batch, device)
    if "padding_mask" not in moved or moved["padding_mask"] is None:
        raise KeyError("batch missing padding_mask; export_eval refuses fail-open padding semantics")
    mode = str(oer_mode or "none").strip().lower()
    if mode not in {"none", "fp32_oracle", "current"}:
        raise ValueError(f"unknown OER export mode: {oer_mode!r}")
    with torch.inference_mode():
        encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
            model,
            moved["waveform"],
            moved["padding_mask"],
        )
        manager: TokenTraceHookManager | None = None
        oer_manager: _OerInputHookManager | None = None
        if token_layer_indices:
            manager = TokenTraceHookManager(
                model,
                layer_indices=tuple(int(item) for item in token_layer_indices),
                positions=("block_output",),
                require_padding_mask=True,
            )
            manager.set_batch_context(
                sample_id=[str(item) for item in sample_ids],
                padding_mask=token_padding_mask.detach().cpu(),
                clear_cache=True,
            )
            manager.register_hooks()
        if mode == "fp32_oracle":
            oer_manager = _OerInputHookManager(model, oer_hook_list, token_padding_mask)
            oer_manager.register_hooks()
        elif mode == "current":
            _clear_quant_oer_state(model, oer_hook_list)
        try:
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            features = features.float()
            records = [] if manager is None else manager.pop_cache()
            if mode == "fp32_oracle":
                oer_activations = {} if oer_manager is None else oer_manager.pop_cache()
            elif mode == "current":
                oer_activations = _collect_quant_oer_activations(model, oer_hook_list, token_padding_mask)
            else:
                oer_activations = {}
        finally:
            if manager is not None:
                manager.clear_hooks()
                manager.clear_batch_context()
            if oer_manager is not None:
                oer_manager.clear_hooks()
        embedding, valid_token_count = mask_aware_mean(features, token_padding_mask)
        logits = pooled_logits_from_features(model, features, token_padding_mask)
    _assert_finite_tensor("logits", logits)
    _assert_finite_tensor("embeddings", embedding)
    return logits.float(), embedding.float(), token_padding_mask, valid_token_count, records, oer_activations

def load_trained_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    qparams_csv_override: Path | None = None,
) -> tuple[BEATs, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"trained checkpoint not found for export_eval: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"trained checkpoint must be a mapping, got {type(checkpoint)!r}")
    model_kind = _infer_checkpoint_model_kind(checkpoint)
    if model_kind == "fp32_teacher":
        _require_checkpoint_keys(checkpoint, ("cfg", "model", "label_mapping"), model_kind=model_kind)
        label_mapping = _normalise_checkpoint_label_mapping(checkpoint.get("label_mapping"))
        model = _build_finetuned_beats(checkpoint["cfg"], label_mapping)
        state = checkpoint["model"]
        if not isinstance(state, Mapping):
            raise TypeError(f"checkpoint['model'] must be a state_dict mapping, got {type(state)!r}")
        model.load_state_dict(state, strict=True)
        quant_audit = audit_quant_state(model, checkpoint=checkpoint, expected_quantized_layers=[])
    else:
        _require_checkpoint_keys(
            checkpoint,
            ("cfg", "model_state_dict", "label_mapping", "quantized_layers", "qparams", "qparams_hash", "qparams_csv"),
            model_kind=model_kind,
        )
        label_mapping = _normalise_checkpoint_label_mapping(checkpoint.get("label_mapping"))
        model = _build_finetuned_beats(checkpoint["cfg"], label_mapping)
        quantized_layers = _checkpoint_quantized_layers(checkpoint, model_kind=model_kind)
        qparams_payload, computed_qparams_hash, _qparams_csv = _verify_checkpoint_qparams(
            checkpoint,
            checkpoint_path,
            qparams_csv_override=qparams_csv_override,
        )
        controller = prepare_beats_ffn_int4_quantization_from_project_config(model)
        _apply_checkpoint_qparams(controller, qparams_payload)
        state = checkpoint["model_state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError(f"checkpoint['model_state_dict'] must be a state_dict mapping, got {type(state)!r}")
        model.load_state_dict(state, strict=True)
        controller.qparams_frozen = True
        controller.set_observer_enabled(False)
        controller.set_fake_quant_enabled(True)
        quant_audit = audit_quant_state(
            model,
            checkpoint=checkpoint,
            quant_controller=controller,
            expected_quantized_layers=quantized_layers,
            expected_qparams_hash=computed_qparams_hash,
        )
        setattr(model, "_baseline_quant_controller", controller)
    model.to(device)
    model.eval()
    _assert_no_parameter_grads(model, context=f"load_trained_checkpoint({checkpoint_path})")
    reference_checkpoint_hash = ""
    if model_kind == REFERENCE_MODEL_NAME:
        reference_checkpoint_hash = file_sha256(checkpoint_path) or ""
    else:
        reference_checkpoint_hash = str(checkpoint.get("teacher_checkpoint_sha256") or checkpoint.get("teacher_checkpoint_hash") or "")
    setattr(model, "_baseline_model_kind", model_kind)
    setattr(model, "_baseline_reference_checkpoint_hash", reference_checkpoint_hash)
    setattr(model, "_baseline_checkpoint_quant_identity", {"quantized_layers": checkpoint.get("quantized_layers", []), "qparams_hash": checkpoint.get("qparams_hash")})
    setattr(model, "_baseline_quant_audit", quant_audit)
    payload = dict(checkpoint)
    payload["_baseline_model_kind"] = model_kind
    payload["_baseline_quant_audit"] = quant_audit
    return model, payload

def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    raw = batch.get("sample_id")
    if raw is None:
        raise KeyError("batch missing sample_id; export_eval requires stable sample order")
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    return [str(raw)]


def _buckets_for_labels(labels: Tensor, label_buckets: Mapping[int, str]) -> list[str]:
    buckets: list[str] = []
    missing: list[int] = []
    for label_id in labels.detach().cpu().long().reshape(-1).tolist():
        label_key = int(label_id)
        bucket = label_buckets.get(label_key)
        if bucket is None:
            missing.append(label_key)
            continue
        buckets.append(str(bucket))
    if missing:
        raise ValueError(f"missing canonical bucket for label_id values: {sorted(set(missing))[:8]}")
    return buckets


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _tensor_nbytes(tensor: Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _tensor_chunks_nbytes(chunks: Sequence[Tensor]) -> int:
    return sum(_tensor_nbytes(chunk) for chunk in chunks)


def _bytes_to_gib(value: int) -> float:
    return float(value) / float(1024 ** 3)


def _max_export_payload_gib(args: argparse.Namespace) -> float:
    value = getattr(args, "max_export_payload_gib", DEFAULT_MAX_EXPORT_PAYLOAD_GIB)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--max-export-payload-gib must be a finite positive float, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"--max-export-payload-gib must be a finite positive float, got {value!r}")
    return parsed


def _resolve_payload_checkpoint_path(paths: Any, checkpoint_path: str | Path | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    return Path(paths.output_checkpoint)


def _export_dirs(export_root: str | Path | None, export_name: str) -> dict[str, Path]:
    root = Path(export_root) if export_root is not None else Path(project_config.BASE_DIR) / "outputs"
    name = str(export_name).strip() or "fp32_teacher"
    dirs = {
        "logits": root / "logits" / name,
        "embeddings": root / "embeddings" / name,
        "features": root / "features" / name,
        "predictions": root / "predictions" / name,
        "metrics": root / "metrics" / name,
        "manifests": root / "manifests" / name,
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _path_str(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve(strict=False))


def _prediction_rows(
    *,
    sample_ids: Sequence[str],
    audio_paths: Sequence[str],
    labels: Tensor,
    predictions: Tensor,
    logits: Tensor,
    buckets: Sequence[str],
) -> list[dict[str, Any]]:
    labels = labels.detach().cpu().long().reshape(-1)
    predictions = predictions.detach().cpu().long().reshape(-1)
    logits = logits.detach().cpu().float()
    margins = correct_logit_margin(logits, labels).detach().cpu().float().reshape(-1)
    gaps = top1_top2_gap(logits).detach().cpu().float().reshape(-1)
    if not (len(sample_ids) == len(audio_paths) == len(buckets) == int(labels.numel()) == int(predictions.numel())):
        raise ValueError("prediction row inputs must share the same sample dimension")
    rows: list[dict[str, Any]] = []
    tail_flags = [bool(item) for item in tail_included(list(buckets))]
    for index, sample_id in enumerate(sample_ids):
        label_id = int(labels[index].item())
        prediction = int(predictions[index].item())
        rows.append(
            {
                "sample_index": int(index),
                "sample_id": str(sample_id),
                "audio_path": str(audio_paths[index]),
                "label_id": label_id,
                "prediction": prediction,
                "correct": bool(prediction == label_id),
                "bucket": str(buckets[index]),
                "tail_included": bool(tail_flags[index]),
                "correct_logit_margin": float(margins[index].item()),
                "top1_top2_gap": float(gaps[index].item()),
            }
        )
    return rows


def _model_num_classes(model: nn.Module) -> int:
    predictor = getattr(model, "predictor", None)
    out_features = getattr(predictor, "out_features", None)
    if out_features is not None:
        return int(out_features)
    return int(getattr(getattr(model, "cfg", object()), "predictor_class"))


def _conv_output_size(size: int, *, kernel: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    return int(math.floor((int(size) + 2 * int(padding) - int(dilation) * (int(kernel) - 1) - 1) / int(stride) + 1))


def _estimate_fbank_frames(audio_cfg: AudioInputConfig) -> int:
    sample_rate = int(audio_cfg.sample_rate)
    samples = int(audio_cfg.target_num_samples)
    frame_length = int(round(sample_rate * 0.025))
    frame_shift = int(round(sample_rate * 0.010))
    if samples < frame_length:
        raise RuntimeError(f"cannot estimate BEATs fbank frames: target samples {samples} < frame length {frame_length}")
    return 1 + (samples - frame_length) // frame_shift


def _estimate_token_count(model: BEATs, audio_cfg: AudioInputConfig) -> int:
    conv = getattr(model, "patch_embedding", None)
    if conv is None:
        raise RuntimeError("cannot estimate token count: model lacks patch_embedding")
    kernel = tuple(int(item) for item in getattr(conv, "kernel_size", ()))
    stride = tuple(int(item) for item in getattr(conv, "stride", ()))
    padding = tuple(int(item) for item in getattr(conv, "padding", (0, 0)))
    dilation = tuple(int(item) for item in getattr(conv, "dilation", (1, 1)))
    if len(kernel) != 2 or len(stride) != 2:
        raise RuntimeError(f"cannot estimate token count from patch_embedding kernel={kernel}, stride={stride}")
    frames = _estimate_fbank_frames(audio_cfg)
    mel_bins = 128
    out_time = _conv_output_size(frames, kernel=kernel[0], stride=stride[0], padding=padding[0], dilation=dilation[0])
    out_mel = _conv_output_size(mel_bins, kernel=kernel[1], stride=stride[1], padding=padding[1], dilation=dilation[1])
    if out_time <= 0 or out_mel <= 0:
        raise RuntimeError(f"invalid estimated BEATs token grid: frames={frames}, mel_bins={mel_bins}, out=({out_time},{out_mel})")
    return int(out_time * out_mel)


def _estimate_export_capacity(model: BEATs, loader: DataLoader, token_layer_indices: Sequence[int]) -> dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "__len__"):
        raise RuntimeError("export_eval capacity estimate requires a DataLoader dataset with __len__")
    audio_cfg = audio_config_from_project()
    num_samples = int(len(dataset))
    num_layers = int(len(token_layer_indices))
    num_tokens = _estimate_token_count(model, audio_cfg)
    hidden_dim = int(getattr(model.cfg, "encoder_embed_dim"))
    num_classes = _model_num_classes(model)
    dtype_bytes = 4
    estimated_token_bytes = int(num_samples * num_layers * num_tokens * hidden_dim * dtype_bytes)
    estimated_logits_bytes = int(num_samples * num_classes * dtype_bytes)
    estimated_embeddings_bytes = int(num_samples * hidden_dim * dtype_bytes)
    estimated_masks_bytes = int(num_samples * num_tokens)
    ffn_dim = int(getattr(model.cfg, "encoder_ffn_embed_dim", hidden_dim * 4))
    fc1_topk = max(1, int(math.ceil(float(num_tokens * hidden_dim) * float(OER_TOPK_RATIO))))
    fc2_topk = max(1, int(math.ceil(float(num_tokens * ffn_dim) * float(OER_TOPK_RATIO))))
    estimated_oer_oracle_bytes = int(num_samples * OER_EXPECTED_NUM_LAYERS * (fc1_topk + fc2_topk) * 8)
    estimated_total_bytes = int(estimated_token_bytes + estimated_logits_bytes + estimated_embeddings_bytes + estimated_masks_bytes + estimated_oer_oracle_bytes)
    return {
        "num_samples": num_samples,
        "num_token_layers": num_layers,
        "estimated_num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "export_dtype": "float32",
        "dtype_bytes": dtype_bytes,
        "estimated_token_bytes": estimated_token_bytes,
        "estimated_token_gib": _bytes_to_gib(estimated_token_bytes),
        "estimated_logits_bytes": estimated_logits_bytes,
        "estimated_embeddings_bytes": estimated_embeddings_bytes,
        "estimated_masks_bytes": estimated_masks_bytes,
        "estimated_oer_oracle_bytes": estimated_oer_oracle_bytes,
        "estimated_oer_oracle_gib": _bytes_to_gib(estimated_oer_oracle_bytes),
        "estimated_total_bytes": estimated_total_bytes,
        "estimated_total_gib": _bytes_to_gib(estimated_total_bytes),
    }


def _ensure_disk_capacity(root: Path, estimated_bytes: int) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(root.resolve(strict=False)))
    required = int(math.ceil(float(estimated_bytes) * 1.05)) + 64 * 1024 * 1024
    if int(usage.free) < required:
        raise RuntimeError(
            f"insufficient disk space for export artifacts under {root}: "
            f"free={int(usage.free)} bytes, required={required} bytes"
        )
    return {
        "disk_root": _path_str(root),
        "disk_free_bytes_before_export": int(usage.free),
        "disk_required_bytes_with_margin": int(required),
    }


def _token_export_mode_from_layers(model: nn.Module, token_layer_indices: Sequence[int]) -> str:
    if not token_layer_indices:
        return "none"
    all_layers = resolve_full_layer_indices(model)
    return "all" if tuple(int(item) for item in token_layer_indices) == all_layers else "sparse"


def _token_feature_root(output_dirs: Mapping[str, Path], *, split: str, flat_output: bool) -> Path | None:
    root = Path(output_dirs["features"])
    return root / "token_features" if flat_output else root / f"{split}_token_features"


def _write_token_feature_shard(
    *,
    token_root: Path,
    model_name: str,
    split: str,
    seed: int,
    layer_idx: int,
    shard_id: int,
    start_row: int,
    end_row: int,
    record: Any,
    run_identity_hash: str,
    run_manifest_path: Path,
) -> dict[str, Any]:
    tensor = record.tensor.detach().cpu().float().contiguous()
    padding_mask = record.padding_mask.detach().cpu().bool().contiguous()
    valid_mask = record.valid_token_mask.detach().cpu().bool().contiguous()
    _assert_finite_tensor(f"token_features.layer_{int(layer_idx)}.shard_{int(shard_id)}", tensor, valid_token_mask=valid_mask)
    layer_dir = token_root / f"layer_{int(layer_idx):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    shard_path = layer_dir / f"shard_{int(shard_id):05d}.pt"
    payload = {
        "artifact_type": "baseline_token_feature_shard",
        "artifact_kind": "token_feature_shard",
        "artifact_protocol_version": "baseline_token_shard.v1",
        "run_identity_hash": str(run_identity_hash),
        "run_manifest_path": _path_str(run_manifest_path),
        "model_name": str(model_name),
        "split": str(split),
        "seed": int(seed),
        "layer": int(layer_idx),
        "shard_id": int(shard_id),
        "start_row": int(start_row),
        "end_row": int(end_row),
        "row_interval": "[start_row,end_row)",
        "sample_id": [str(item) for item in record.sample_id],
        "tensor": tensor,
        "token_feature": tensor,
        "token_padding_mask": padding_mask,
        "valid_token_mask": valid_mask,
        "tensor_semantics": "Transformer encoder block output before pooling and before classifier head; padding tokens retained",
        "tensor_position": "encoder_block_output",
        "tensor_layout": "[B,T,C]",
        "original_layout": str(record.original_layout),
        "original_shape": tuple(int(item) for item in record.original_shape),
        "canonical_layout": str(record.canonical_layout),
        "canonical_shape": tuple(int(item) for item in record.canonical_shape),
        "layout_inference_source": str(record.layout_inference_source),
        "padding_mask_available": bool(record.padding_mask_available),
        "padding_mask_aligned": bool(record.padding_mask_aligned),
        "mask_status": str(record.mask_status),
    }
    torch.save(payload, shard_path)
    file_hash = file_sha256(shard_path)
    return {
        "model_name": str(model_name),
        "split": str(split),
        "seed": int(seed),
        "run_identity_hash": str(run_identity_hash),
        "run_manifest_path": _path_str(run_manifest_path),
        "artifact_kind": "token_feature_shard",
        "layer": int(layer_idx),
        "shard_id": int(shard_id),
        "start_row": int(start_row),
        "end_row": int(end_row),
        "sample_id_start": str(record.sample_id[0]) if record.sample_id else "",
        "sample_id_end": str(record.sample_id[-1]) if record.sample_id else "",
        "shape": json.dumps([int(item) for item in tensor.shape], separators=(",", ":")),
        "dtype": "float32",
        "file_path": _path_str(shard_path),
        "file_sha256": file_hash,
    }


def _artifact_hashes(paths: Mapping[str, Any], token_index_rows: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name, value in paths.items():
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            target = Path(value)
            if target.exists() and target.is_file():
                result[str(target.resolve(strict=False))] = file_sha256(target)
    for row in token_index_rows:
        file_path = row.get("file_path")
        if file_path:
            result[str(file_path)] = row.get("file_sha256")
    return result



def _load_tensor_payload(path: Path) -> Mapping[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"required FP32 reference artifact missing: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"FP32 reference artifact must be a mapping, got {type(payload)!r}: {path}")
    return payload


def _payload_first(payload: Mapping[str, Any], keys: Sequence[str], *, path: Path) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"{path} missing required reference field; expected one of {list(keys)}")


def _payload_sample_ids(payload: Mapping[str, Any], *, path: Path) -> list[str]:
    raw = _payload_first(payload, ("sample_ids", "sample_id"), path=path)
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    raise TypeError(f"{path} reference sample_ids must be a sequence, got {type(raw)!r}")


def _payload_labels(payload: Mapping[str, Any], *, path: Path) -> Tensor:
    raw = _payload_first(payload, ("labels", "label_id"), path=path)
    labels = torch.as_tensor(raw).detach().cpu().reshape(-1)
    if labels.numel() <= 0:
        raise ValueError(f"{path} reference labels are empty")
    return labels


def _payload_tensor(payload: Mapping[str, Any], keys: Sequence[str], *, path: Path, name: str) -> Tensor:
    tensor = torch.as_tensor(_payload_first(payload, keys, path=path)).detach().cpu().float().contiguous()
    if tensor.ndim != 2 or int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0:
        raise ValueError(f"{path} reference {name} must be [N,D/C], got shape={tuple(tensor.shape)}")
    _assert_finite_tensor(f"reference.{name}", tensor)
    return tensor


def _reference_artifact_paths(reference_root: str | Path, *, split: str) -> tuple[Path, Path, Path]:
    root = Path(reference_root)
    nonflat_logits = root / "logits" / REFERENCE_MODEL_NAME / f"{split}_logits.pt"
    nonflat_embeddings = root / "embeddings" / REFERENCE_MODEL_NAME / f"{split}_embeddings.pt"
    nonflat_manifest = root / "manifests" / REFERENCE_MODEL_NAME / "run_manifest.json"
    if nonflat_logits.exists() or nonflat_embeddings.exists() or nonflat_manifest.exists():
        return nonflat_logits, nonflat_embeddings, nonflat_manifest
    return root / "logits.pt", root / "embeddings.pt", root / "run_manifest.json"


def _load_reference_manifest(path: Path, *, split: str) -> Mapping[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"FP32 reference manifest missing for split={split}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"FP32 reference manifest must contain a JSON object: {path}")
    manifest_model = str(payload.get("model_kind") or payload.get("model_name") or "")
    if manifest_model != REFERENCE_MODEL_NAME:
        raise ValueError(f"FP32 reference manifest must be model_kind/model_name={REFERENCE_MODEL_NAME!r}: {path}")
    checkpoint_hash = str(payload.get("checkpoint_sha256") or "")
    if not checkpoint_hash:
        raise ValueError(f"FP32 reference manifest missing checkpoint_sha256: {path}")
    split_hashes = payload.get("split_hashes")
    if isinstance(split_hashes, Mapping) and not str(split_hashes.get(split) or ""):
        raise ValueError(f"FP32 reference manifest missing split_hashes[{split!r}]: {path}")
    return payload


def _manifest_expected_artifact_hash(manifest: Mapping[str, Any], artifact_path: Path) -> str:
    artifact_sha256 = manifest.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping):
        raise ValueError("FP32 reference manifest missing artifact_sha256 mapping")
    resolved = str(artifact_path.resolve(strict=False))
    for key in (resolved, str(artifact_path), artifact_path.name):
        value = artifact_sha256.get(key)
        if value:
            return str(value)

    def unique_match(matches: Sequence[tuple[str, Any]], *, match_kind: str) -> str | None:
        values = sorted({str(value) for _key, value in matches if value})
        if len(values) == 1:
            return values[0]
        if len(values) > 1:
            keys = [str(key) for key, _value in matches[:6]]
            raise ValueError(
                f"FP32 reference manifest artifact_sha256 has ambiguous {match_kind} matches "
                f"for {resolved}: {keys}"
            )
        return None

    artifact_parts = _portable_path_parts(resolved)
    for width in range(min(8, len(artifact_parts)), 1, -1):
        suffix = tuple(artifact_parts[-width:])
        matches = [
            (str(key), value)
            for key, value in artifact_sha256.items()
            if value
            and len(_portable_path_parts(str(key))) >= width
            and tuple(_portable_path_parts(str(key))[-width:]) == suffix
        ]
        matched = unique_match(matches, match_kind=f"path suffix {suffix!r}")
        if matched:
            return matched

    filename = _cross_platform_path_name(str(artifact_path))
    matches = [
        (str(key), value)
        for key, value in artifact_sha256.items()
        if value and _cross_platform_path_name(str(key)) == filename
    ]
    matched = unique_match(matches, match_kind=f"filename {filename!r}")
    if matched:
        return matched
    raise ValueError(f"FP32 reference manifest artifact_sha256 does not register {resolved}")


def _assert_manifest_artifact_hash(manifest: Mapping[str, Any], artifact_path: Path) -> None:
    expected = _manifest_expected_artifact_hash(manifest, artifact_path)
    actual = file_sha256(artifact_path)
    if actual != expected:
        raise ValueError(f"FP32 reference artifact hash mismatch for {artifact_path}: manifest={expected} actual={actual}")


def _reference_root_from_args(args: argparse.Namespace, output_dirs: Mapping[str, Path]) -> Path:
    explicit = getattr(args, "reference_root", None)
    if explicit is not None and str(explicit).strip():
        return Path(explicit)
    baseline_root = getattr(args, "baseline_root", None)
    if baseline_root is not None and str(baseline_root).strip():
        return Path(baseline_root) / f"seed_{int(args.seed)}"
    export_root = getattr(args, "export_root", None)
    if export_root is not None and str(export_root).strip():
        return Path(export_root)
    logits_dir = Path(output_dirs["logits"])
    export_name = str(getattr(args, "export_name", "") or "")
    if logits_dir.name == export_name and logits_dir.parent.name == "logits":
        return logits_dir.parent.parent
    raise ValueError("cannot infer FP32 reference root; pass reference_root or baseline_root for INT4/KD export")


def load_fp32_reference_bundle(reference_root: str | Path, *, split: str) -> ReferenceBundle:
    logits_path, embeddings_path, manifest_path = _reference_artifact_paths(reference_root, split=split)
    manifest = _load_reference_manifest(manifest_path, split=split)
    _assert_manifest_artifact_hash(manifest, logits_path)
    _assert_manifest_artifact_hash(manifest, embeddings_path)
    logits_payload = _load_tensor_payload(logits_path)
    embeddings_payload = _load_tensor_payload(embeddings_path)
    for artifact_path, payload in ((logits_path, logits_payload), (embeddings_path, embeddings_payload)):
        if payload.get("reference_ready") is not True:
            raise ValueError(f"FP32 artifact is not reference-ready: {artifact_path}")
        if str(payload.get("model_kind") or "") != REFERENCE_MODEL_NAME:
            raise ValueError(f"FP32 reference artifact model_kind mismatch: {artifact_path}")
        if str(payload.get("reference_artifact_protocol_version") or "") != REFERENCE_PROTOCOL_VERSION:
            raise ValueError(f"FP32 reference artifact protocol mismatch: {artifact_path}")
    sample_ids = _payload_sample_ids(logits_payload, path=logits_path)
    embedding_sample_ids = _payload_sample_ids(embeddings_payload, path=embeddings_path)
    if sample_ids != embedding_sample_ids:
        raise ValueError("FP32 reference logits and embeddings sample order differ")
    labels = _payload_labels(logits_payload, path=logits_path)
    embedding_labels = _payload_labels(embeddings_payload, path=embeddings_path)
    if tuple(labels.shape) != tuple(embedding_labels.shape) or not bool(torch.equal(labels, embedding_labels)):
        raise ValueError("FP32 reference logits and embeddings labels differ")
    logits = _payload_tensor(logits_payload, ("logits",), path=logits_path, name="logits")
    embeddings = _payload_tensor(embeddings_payload, ("embeddings", "embedding"), path=embeddings_path, name="embeddings")
    if int(logits.shape[0]) != len(sample_ids) or int(embeddings.shape[0]) != len(sample_ids):
        raise ValueError("FP32 reference artifact sample dimension differs from sample_ids length")
    identity_keys = ("sample_order_hash", "label_mapping_hash", "split_hash", "pooling", "run_identity_hash")
    for key in identity_keys:
        left = str(logits_payload.get(key) or "")
        right = str(embeddings_payload.get(key) or "")
        if not left or not right or left != right:
            raise ValueError(f"FP32 reference logits/embeddings identity field {key!r} is missing or mismatched")
    manifest_split_hashes = manifest.get("split_hashes")
    manifest_split_hash = ""
    if isinstance(manifest_split_hashes, Mapping):
        manifest_split_hash = str(manifest_split_hashes.get(split) or "")
    if manifest_split_hash and manifest_split_hash != str(logits_payload.get("split_hash") or ""):
        raise ValueError("FP32 reference artifact split_hash differs from manifest split_hashes entry")
    manifest_label_mapping_hash = str(manifest.get("label_mapping_hash") or "")
    if manifest_label_mapping_hash and manifest_label_mapping_hash != str(logits_payload.get("label_mapping_hash") or ""):
        raise ValueError("FP32 reference artifact label_mapping_hash differs from manifest")
    return ReferenceBundle(
        embeddings=embeddings,
        logits=logits,
        sample_ids=sample_ids,
        labels=labels,
        label_mapping_hash=str(logits_payload.get("label_mapping_hash") or ""),
        split_hash=str(logits_payload.get("split_hash") or ""),
        sample_order_hash=str(logits_payload.get("sample_order_hash") or ""),
        pooling=str(logits_payload.get("pooling") or ""),
        model_kind=REFERENCE_MODEL_NAME,
        checkpoint_hash=str(manifest.get("checkpoint_sha256") or ""),
        embedding_dim=int(embeddings.shape[1]),
    )


def _reference_bundle_for_export(
    *,
    args: argparse.Namespace,
    output_dirs: Mapping[str, Path],
    split: str,
    model_kind: str,
    model: nn.Module,
) -> ReferenceBundle | None:
    if model_kind == REFERENCE_MODEL_NAME:
        return None
    if model_kind not in {
        "int4_student_init",
        "vanilla_kd",
        "int4_student",
        "two_stage_decoupling",
        "two_stage_class_prior_tau",
        "two_stage_tail_repair",
    }:
        return None
    reference_root = _reference_root_from_args(args, output_dirs)
    reference = load_fp32_reference_bundle(reference_root, split=split)
    expected_hash = str(getattr(model, "_baseline_reference_checkpoint_hash", "") or "")
    if expected_hash and expected_hash != reference.checkpoint_hash:
        raise ValueError(
            f"{model_kind} checkpoint expects FP32 reference checkpoint {expected_hash}, "
            f"but loaded reference artifact uses {reference.checkpoint_hash}"
        )
    return reference
def export_eval_split(
    *,
    model: BEATs,
    paths: Any,
    split: str,
    args: argparse.Namespace,
    label_buckets: Mapping[int, str],
    token_layer_indices: Sequence[int],
    output_dirs: Mapping[str, Path],
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    quant_controller: Any | None = None,
    quant_audit: Mapping[str, Any] | None = None,
    run_identity_hash: str | None = None,
) -> dict[str, Any]:
    if bool(getattr(args, "amp", False)):
        raise RuntimeError("export_eval_split forbids AMP; pass a formal FP32 export args object")
    model.eval()
    _assert_no_parameter_grads(model, context=f"export_eval_split({split})")
    run_identity_hash = str(run_identity_hash or getattr(args, "run_identity_hash", "") or "")
    if not run_identity_hash:
        raise ValueError("export_eval_split requires run_identity_hash before writing artifacts")
    export_name = str(getattr(args, "export_name", "") or "").strip() or "baseline"
    model_kind = str(
        getattr(model, "_baseline_model_kind", "")
        or getattr(args, "model_kind", "")
        or export_name
    ).strip()
    if not model_kind:
        raise ValueError("export_eval_split requires model_kind on the loaded model or args")
    payload_checkpoint_path = _resolve_payload_checkpoint_path(paths, checkpoint_path)
    payload_checkpoint_sha256 = file_sha256(payload_checkpoint_path)
    if payload_checkpoint_sha256 is None:
        raise FileNotFoundError(f"export_eval checkpoint_path is missing or not hashable: {payload_checkpoint_path}")
    label_mapping_path = Path(getattr(paths, "label_mapping"))
    label_mapping = _load_label_mapping_json(label_mapping_path)
    label_mapping_hash = file_sha256(label_mapping_path)
    if label_mapping_hash is None:
        raise FileNotFoundError(f"export_eval label_mapping is missing or not hashable: {label_mapping_path}")
    current_split_csv = _split_csv_path(paths, split)
    current_split_hash = file_sha256(current_split_csv)
    if current_split_hash is None:
        raise FileNotFoundError(f"export_eval split CSV is missing or not hashable: {current_split_csv}")
    checkpoint_quant_identity = getattr(model, "_baseline_checkpoint_quant_identity", None)
    if quant_controller is not None and isinstance(checkpoint_quant_identity, Mapping):
        active_quant_audit = audit_quant_state(
            model,
            checkpoint=checkpoint_quant_identity,
            quant_controller=quant_controller,
            expected_qparams_hash=checkpoint_quant_identity.get("qparams_hash"),
        )
    else:
        active_quant_audit = dict(quant_audit or getattr(model, "_baseline_quant_audit", {}) or {})
        if not active_quant_audit:
            active_quant_audit = audit_quant_state(model, quant_controller=quant_controller)

    loader = build_loader(
        paths,
        split=split,
        batch_size=int(args.eval_batch_size or args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    flat_output = bool(getattr(args, "flat_output", False))
    token_export_mode = _token_export_mode_from_layers(model, token_layer_indices)
    capacity = _estimate_export_capacity(model, loader, token_layer_indices)
    max_payload_gib = _max_export_payload_gib(args)
    capacity["max_export_payload_gib"] = float(max_payload_gib)
    capacity["estimated_token_exceeds_threshold"] = bool(capacity["estimated_token_gib"] > max_payload_gib)
    capacity["token_over_threshold_policy"] = "write_sharded_token_features" if capacity["estimated_token_exceeds_threshold"] else "single_pass_sharded_token_features"
    disk = _ensure_disk_capacity(Path(output_dirs["features"]), int(capacity["estimated_total_bytes"]))


    oer_hook_list = _baseline_oer_hook_list(model)
    oer_hook_list_hash = _oer_hook_list_hash(oer_hook_list)
    if model_kind == REFERENCE_MODEL_NAME:
        oer_mode = "fp32_oracle"
    elif model_kind in {"int4_student_init", "int4_student", "vanilla_kd"}:
        oer_mode = "current"
    else:
        oer_mode = "none"
    oer_oracle_writer: _OerOracleWriter | None = None
    oer_oracle_reader: _OerOracleReader | None = None
    oer_oracle_artifact: dict[str, Any] | None = None
    oer_reference_metadata: dict[str, Any] | None = None
    if oer_mode == "fp32_oracle":
        oer_oracle_writer = _OerOracleWriter(
            root=_oer_oracle_root(output_dirs, split=split, flat_output=flat_output),
            split=split,
            seed=int(args.seed),
            model_name=export_name,
            run_identity_hash=run_identity_hash,
            run_manifest_path=Path(output_dirs["manifests"]) / "run_manifest.json",
            checkpoint_path=payload_checkpoint_path,
            checkpoint_sha256=payload_checkpoint_sha256,
            label_mapping_hash=label_mapping_hash,
            split_hash=current_split_hash,
            hook_list=oer_hook_list,
        )
    elif oer_mode == "current":
        reference_root = _reference_root_from_args(args, output_dirs)
        oracle_path, oracle_manifest_path = _reference_oer_oracle_paths(reference_root, split=split)
        expected_reference_hash = str(getattr(model, "_baseline_reference_checkpoint_hash", "") or "")
        oer_oracle_reader = _OerOracleReader(
            oracle_path=oracle_path,
            manifest_path=oracle_manifest_path,
            split=split,
            hook_list=oer_hook_list,
            label_mapping_hash=label_mapping_hash,
            split_hash=current_split_hash,
            expected_checkpoint_hash=expected_reference_hash,
        )
        oer_reference_metadata = {
            "oracle_path": _path_str(oracle_path),
            "oracle_manifest_path": _path_str(oracle_manifest_path),
            "oracle_sha256": file_sha256(oracle_path),
            "oracle_manifest_sha256": file_sha256(oracle_manifest_path),
            "oracle_source_checkpoint_sha256": oer_oracle_reader.manifest.get("oracle_source_checkpoint_sha256"),
        }
    sample_ids: list[str] = []
    audio_paths: list[str] = []
    buckets: list[str] = []
    label_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []
    embedding_chunks: list[Tensor] = []
    padding_mask_chunks: list[Tensor] = []
    valid_count_chunks: list[Tensor] = []
    token_index_rows: list[dict[str, Any]] = []
    oer_records: list[dict[str, Any]] = []

    token_root = _token_feature_root(output_dirs, split=split, flat_output=flat_output) if token_layer_indices else None
    run_manifest_path = Path(output_dirs["manifests"]) / "run_manifest.json"
    seen: set[str] = set()
    for batch_index, batch in enumerate(loader):
        batch_sample_ids = _batch_sample_ids(batch)
        duplicates = sorted(set(batch_sample_ids) & seen)
        if duplicates:
            raise ValueError(f"duplicate sample_id values during {split} export: {duplicates[:8]}")
        seen.update(batch_sample_ids)
        labels = batch["label_id"].detach().cpu().long().reshape(-1)
        if int(labels.numel()) != len(batch_sample_ids):
            raise ValueError(f"{split} batch label count differs from sample_id count")
        batch_buckets = _buckets_for_labels(labels, label_buckets)
        start_row = int(len(sample_ids))
        end_row = int(start_row + len(batch_sample_ids))
        logits, embeddings, token_padding_mask, valid_token_count, records, oer_activations = forward_export_batch(
            model,
            batch,
            sample_ids=batch_sample_ids,
            token_layer_indices=token_layer_indices,
            device=device,
            oer_mode=oer_mode,
            oer_hook_list=oer_hook_list if oer_mode != "none" else (),
        )
        if oer_oracle_writer is not None:
            oer_oracle_writer.write_shard(
                shard_id=int(batch_index),
                start_row=start_row,
                end_row=end_row,
                sample_ids=batch_sample_ids,
                activations=oer_activations,
            )
        if oer_oracle_reader is not None:
            oer_records.extend(
                oer_oracle_reader.records_for_batch(
                    sample_ids=batch_sample_ids,
                    labels=labels,
                    buckets=batch_buckets,
                    activations=oer_activations,
                )
            )
        record_by_layer = {int(record.layer_idx): record for record in records}
        missing_layers = [int(layer_idx) for layer_idx in token_layer_indices if int(layer_idx) not in record_by_layer]
        if missing_layers:
            raise RuntimeError(f"missing token trace records for layers {missing_layers} in {split} batch {batch_index}")
        if token_root is not None:
            for layer_idx in token_layer_indices:
                token_index_rows.append(
                    _write_token_feature_shard(
                        token_root=token_root,
                        model_name=str(getattr(args, "export_name", "baseline")),
                        split=split,
                        seed=int(args.seed),
                        layer_idx=int(layer_idx),
                        shard_id=int(batch_index),
                        start_row=start_row,
                        end_row=end_row,
                        record=record_by_layer[int(layer_idx)],
                        run_identity_hash=run_identity_hash,
                        run_manifest_path=run_manifest_path,
                    )
                )

        sample_ids.extend(batch_sample_ids)
        audio_paths.extend(str(item) for item in batch.get("audio_path", [""] * len(batch_sample_ids)))
        buckets.extend(batch_buckets)
        label_chunks.append(labels.contiguous())
        logits_chunks.append(logits.detach().cpu().float().contiguous())
        embedding_chunks.append(embeddings.detach().cpu().float().contiguous())
        padding_mask_chunks.append(token_padding_mask.detach().cpu().bool().contiguous())
        valid_count_chunks.append(valid_token_count.detach().cpu().long().contiguous())

    if not sample_ids:
        raise RuntimeError(f"{split} loader produced no samples for export_eval")

    label_all = torch.cat(label_chunks, dim=0)
    logits_all = torch.cat(logits_chunks, dim=0).float().contiguous()
    predictions_all = logits_all.argmax(dim=1).cpu().long().contiguous()
    embedding_all = torch.cat(embedding_chunks, dim=0).float().contiguous()
    padding_mask_all = torch.cat(padding_mask_chunks, dim=0).bool().contiguous()
    valid_count_all = torch.cat(valid_count_chunks, dim=0).long().contiguous()
    _assert_finite_tensor("logits_all", logits_all)
    _assert_finite_tensor("embeddings_all", embedding_all)

    sample_order_hash = _sample_order_hash(sample_ids)
    if oer_oracle_writer is not None:
        oer_oracle_artifact = oer_oracle_writer.finish(sample_ids=sample_ids, sample_order_hash=sample_order_hash)
    if oer_oracle_reader is not None:
        oer_oracle_reader.validate_final_sample_order(sample_ids, sample_order_hash)
        if oer_reference_metadata is not None:
            migrations = [
                {"shard_id": int(shard_id), **dict(payload)}
                for shard_id, payload in sorted(oer_oracle_reader.shard_path_migrations.items())
            ]
            oer_reference_metadata["oracle_shard_path_migration_count"] = int(len(migrations))
            oer_reference_metadata["oracle_shard_path_migrations"] = migrations[:16]
    logits_path = output_dirs["logits"] / ("logits.pt" if flat_output else f"{split}_logits.pt")
    embeddings_path = output_dirs["embeddings"] / ("embeddings.pt" if flat_output else f"{split}_embeddings.pt")
    predictions_path = output_dirs["predictions"] / ("predictions.csv" if flat_output else f"{split}_predictions.csv")
    metrics_path = output_dirs["metrics"] / ("metrics.csv" if flat_output else f"{split}_metrics.csv")
    group_metrics_path = output_dirs["metrics"] / ("group_metrics.csv" if flat_output else f"{split}_group_metrics.csv")
    per_class_metrics_path = output_dirs["metrics"] / ("per_class_metrics.csv" if flat_output else f"{split}_per_class_metrics.csv")
    geometry_metrics_path = output_dirs["metrics"] / ("geometry_metrics.csv" if flat_output else f"{split}_geometry_metrics.csv")
    token_index_path = None
    if token_root is not None:
        token_index_path = token_root / "token_features.index.csv"
        _write_csv_rows(token_index_path, token_index_rows)


    checkpoint_hash_available = payload_checkpoint_sha256 is not None
    reference_ready = model_kind == REFERENCE_MODEL_NAME
    reference_identity = {
        "reference_artifact_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "checkpoint_sha256": payload_checkpoint_sha256,
        "sample_order_hash": sample_order_hash,
        "label_mapping_hash": label_mapping_hash,
        "split_hash": current_split_hash,
        "pooling": DEFAULT_POOLING,
        "embedding_dim": int(embedding_all.shape[1]),
        "logit_dim": int(logits_all.shape[1]),
    }
    export_metadata = {
        **capacity,
        **disk,
        "token_export": token_export_mode,
        "export_all_token_layers": token_export_mode == "all",
        "num_samples": int(label_all.numel()),
        "batch_size": int(args.eval_batch_size or args.batch_size),
        "export_memory_mode": "stream_token_shards_accumulate_logits_embeddings",
        "token_feature_protocol": {
            "enabled": bool(token_layer_indices),
            "storage": "sharded_by_layer_and_forward_batch",
            "index_filename": "token_features.index.csv",
            "tensor_semantics": "Transformer encoder block output before pooling and before classifier head; padding tokens retained",
            "tensor_layout": "[B,T,C]",
            "mask_policy": "padding_mask saved per shard; downstream token metrics must exclude padding tokens",
        },
    }
    common_payload = {
        "artifact_protocol_version": "baseline_export_eval.v2",
        "run_identity_hash": run_identity_hash,
        "split": split,
        "dataset_name": paths.name,
        "sample_id": sample_ids,
        "sample_ids": sample_ids,
        "sample_order_hash": sample_order_hash,
        "label_mapping_hash": label_mapping_hash,
        "split_hash": current_split_hash,
        "pooling": DEFAULT_POOLING,
        "model_kind": model_kind,
        "checkpoint_path": _path_str(payload_checkpoint_path),
        "checkpoint_sha256": payload_checkpoint_sha256,
        "reference_ready": bool(reference_ready),
        "reference_artifact_protocol_version": REFERENCE_PROTOCOL_VERSION if reference_ready else "",
        "reference_identity": reference_identity if reference_ready else {},
        "reference_required_fields": ["logits", "embeddings", "sample_ids", "labels", "sample_order_hash", "label_mapping_hash", "split_hash", "pooling"],
        "label_id": label_all,
        "labels": label_all,
        "prediction": predictions_all,
        "bucket": list(buckets),
        "tail_included": [bool(item) for item in tail_included(list(buckets))],
        "audio_path": list(audio_paths),
        "export_name": export_name,
        "oer_protocol_version": OER_PROTOCOL_VERSION,
        "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
        "oer_hook_list_used_hash": oer_hook_list_hash,
    }
    torch.save({**common_payload, "artifact_type": f"{export_name}_logits", "logits": logits_all, "predictions": predictions_all}, logits_path)
    torch.save(
        {
            **common_payload,
            "artifact_type": f"{export_name}_embeddings",
            "embedding": embedding_all,
            "embeddings": embedding_all,
            "embedding_pooling": DEFAULT_POOLING,
            "pooling_definition": POOLING_DEFINITION,
            "token_padding_mask": padding_mask_all,
            "valid_token_count": valid_count_all,
        },
        embeddings_path,
    )

    prediction_rows = _prediction_rows(
        sample_ids=sample_ids,
        audio_paths=audio_paths,
        labels=label_all,
        predictions=predictions_all,
        logits=logits_all,
        buckets=buckets,
    )
    _write_csv_rows(predictions_path, prediction_rows)
    reference_bundle = _reference_bundle_for_export(
        args=args,
        output_dirs=output_dirs,
        split=split,
        model_kind=model_kind,
        model=model,
    )
    reference_metadata = None
    if reference_bundle is not None:
        reference_metadata = {
            "model_kind": reference_bundle.model_kind,
            "checkpoint_hash": reference_bundle.checkpoint_hash,
            "sample_order_hash": reference_bundle.sample_order_hash,
            "label_mapping_hash": reference_bundle.label_mapping_hash,
            "split_hash": reference_bundle.split_hash,
            "pooling": reference_bundle.pooling,
            "embedding_dim": int(reference_bundle.embedding_dim or 0),
        }
    metric_bundle = compute_baseline_metrics(
        sample_ids=sample_ids,
        labels=label_all,
        predictions=predictions_all,
        logits=logits_all,
        embeddings=embedding_all,
        bucket=buckets,
        label_mapping=label_mapping,
        label_buckets=label_buckets,
        split=split,
        model_kind=model_kind,
        seed=int(args.seed),
        run_identity_hash=run_identity_hash,
        label_mapping_hash=label_mapping_hash,
        split_hash=current_split_hash,
        sample_order_hash=sample_order_hash,
        pooling=DEFAULT_POOLING,
        reference=reference_bundle,
        oer_records=oer_records if oer_oracle_reader is not None else None,
    )
    primary_metric_rows = [*metric_bundle["main_metrics"], *metric_bundle.get("retention_metrics", [])]
    _write_csv_rows(metrics_path, primary_metric_rows)
    _write_csv_rows(group_metrics_path, metric_bundle["group_metrics"])
    _write_csv_rows(per_class_metrics_path, metric_bundle["per_class_metrics"])
    _write_csv_rows(geometry_metrics_path, metric_bundle["geometry_metrics"])

    artifact_paths = {
        "logits": _path_str(logits_path),
        "embeddings": _path_str(embeddings_path),
        "predictions": _path_str(predictions_path),
        "metrics": _path_str(metrics_path),
        "group_metrics": _path_str(group_metrics_path),
        "per_class_metrics": _path_str(per_class_metrics_path),
        "geometry_metrics": _path_str(geometry_metrics_path),
        "token_features_root": _path_str(token_root),
        "token_features_index": _path_str(token_index_path),
        "oer_oracle": None if oer_oracle_artifact is None else oer_oracle_artifact.get("path"),
        "oer_oracle_manifest": None if oer_oracle_artifact is None else oer_oracle_artifact.get("manifest_path"),
        "oer_reference_oracle": None if oer_reference_metadata is None else oer_reference_metadata.get("oracle_path"),
        "oer_reference_oracle_manifest": None if oer_reference_metadata is None else oer_reference_metadata.get("oracle_manifest_path"),
    }
    artifact_sha256 = _artifact_hashes(artifact_paths, token_index_rows)
    return {
        "split": split,
        "num_samples": int(label_all.numel()),
        "num_classes_present": int(torch.unique(label_all).numel()),
        "sample_order_hash": sample_order_hash,
        "token_layers": [int(item) for item in token_layer_indices],
        "token_export": token_export_mode,
        "token_feature_shard_count": int(len(token_index_rows)),
        "checkpoint_path": _path_str(payload_checkpoint_path),
        "checkpoint_sha256": payload_checkpoint_sha256,
        "checkpoint_hash_available": bool(checkpoint_hash_available),
        "run_identity_hash": run_identity_hash,
        "reference": reference_metadata,
        "reference_checkpoint_hash": "" if reference_bundle is None else reference_bundle.checkpoint_hash,
        "oer_protocol": {
            "protocol_version": OER_PROTOCOL_VERSION,
            "mode": oer_mode,
            "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
            "oer_hook_list_used": oer_hook_list,
            "oer_hook_list_used_hash": oer_hook_list_hash,
            "oracle_artifact": oer_oracle_artifact,
            "reference_oracle": oer_reference_metadata,
            "sample_record_count": int(len(oer_records)),
        },
        "quant_audit": active_quant_audit,
        **export_metadata,
        "paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
    }


def _normalise_checkpoint_label_mapping_for_compare(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    result = {str(label_name): int(label_id) for label_name, label_id in value.items()}
    _validate_dense_label_mapping(result, context="checkpoint label_mapping")
    return result


def _validate_checkpoint_label_mapping(checkpoint: Mapping[str, Any], label_mapping: Mapping[str, int]) -> None:
    checkpoint_mapping = _normalise_checkpoint_label_mapping_for_compare(checkpoint.get("label_mapping"))
    if checkpoint_mapping is None:
        return
    expected = {str(label_name): int(label_id) for label_name, label_id in label_mapping.items()}
    _validate_dense_label_mapping(expected, context="requested label_mapping")
    if checkpoint_mapping != expected:
        raise ValueError("checkpoint label_mapping differs from label_mapping_path; refusing to export mixed label protocols")


def _run_identity_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_hash({str(key): value for key, value in sorted(payload.items(), key=lambda item: str(item[0]))})


def _split_hash_payload(split_csv: str | Path | None = None, *, paths: Any | None = None) -> dict[str, str | None]:
    if paths is not None:
        return {
            "train": file_sha256(getattr(paths, "train_csv", None)),
            "eval": file_sha256(getattr(paths, "eval_csv", None)),
            "test": file_sha256(getattr(paths, "test_csv", None)),
        }
    return {"split": file_sha256(split_csv)}


def _checkpoint_identity_from_payload(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    model_kind = checkpoint.get("_baseline_model_kind") or _infer_checkpoint_model_kind(checkpoint)
    quantized_layers = [dict(item) for item in checkpoint.get("quantized_layers", [])] if isinstance(checkpoint.get("quantized_layers", []), Sequence) else []
    return {
        "model_kind": str(model_kind),
        "checkpoint_declared_run_identity_hash": checkpoint.get("run_identity_hash"),
        "source_beats_checkpoint_sha256": checkpoint.get("source_beats_checkpoint_sha256") or checkpoint.get("source_beats_checkpoint_hash") or checkpoint.get("source_checkpoint_sha256"),
        "teacher_checkpoint_sha256": checkpoint.get("teacher_checkpoint_sha256") or checkpoint.get("teacher_checkpoint_hash"),
        "qparams_csv_sha256": checkpoint.get("qparams_csv_sha256") or checkpoint.get("qparams_csv_hash"),
        "label_mapping_hash": checkpoint.get("label_mapping_hash"),
        "calibration_hash": checkpoint.get("calibration_hash"),
        "quantized_layers_hash": checkpoint.get("quantized_layers_hash") or canonical_json_hash({"quantized_layers": quantized_layers}),
        "qparams_hash": checkpoint.get("qparams_hash"),
        "bitwidth": checkpoint.get("bitwidth", "fp32" if not quantized_layers else 4),
        "qmin": checkpoint.get("qmin"),
        "qmax": checkpoint.get("qmax"),
        "symmetric": checkpoint.get("symmetric"),
        "weight_granularity": checkpoint.get("weight_granularity"),
        "activation_granularity": checkpoint.get("activation_granularity"),
    }


def build_run_identity(
    *,
    model_name: str,
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    split_hashes: Mapping[str, Any],
    label_mapping_hash: str | None,
    calibration_hash: str | None,
    pooling: str,
    token_export: str,
    token_layers: Sequence[int],
    seed: int,
) -> str:
    identity = {
        "identity_protocol_version": "baseline_run_identity.v1",
        "model_name": str(model_name),
        "checkpoint_path": str(Path(checkpoint_path).resolve(strict=False)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_identity": _checkpoint_identity_from_payload(checkpoint),
        "split_hashes": dict(split_hashes),
        "split_hash": canonical_json_hash(dict(split_hashes)),
        "label_mapping_hash": label_mapping_hash,
        "calibration_hash": calibration_hash or checkpoint.get("calibration_hash"),
        "pooling": str(pooling),
        "token_export": str(token_export),
        "token_layers": [int(item) for item in token_layers],
        "quant_protocol": {
            "quantized_layers_hash": _checkpoint_identity_from_payload(checkpoint).get("quantized_layers_hash"),
            "qparams_hash": checkpoint.get("qparams_hash"),
            "bitwidth": checkpoint.get("bitwidth", "fp32"),
        },
        "oer_protocol": {
            "protocol_version": OER_PROTOCOL_VERSION,
            "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
            "oer_hook_list_used_hash": _oer_hook_list_hash(_expected_oer_hook_list()),
            "topk_ratio": float(OER_TOPK_RATIO),
            "eps": float(OER_EPS),
        },        "audio_input_py_sha256": file_sha256(PROJECT_ROOT / "src" / "data" / "audio_input.py"),
        "group_utils_py_sha256": file_sha256(PROJECT_ROOT / "src" / "utils" / "group_utils.py"),
        "seed": int(seed),
        "git_commit": git_commit(),
    }
    return _run_identity_hash(identity)


def _formal_export_args(args: argparse.Namespace, *, run_identity_hash: str, token_export: str | None = None) -> argparse.Namespace:
    values = dict(vars(args))
    values["amp"] = False
    values["run_identity_hash"] = str(run_identity_hash)
    if token_export is not None:
        values["token_export"] = str(token_export)
    else:
        values.setdefault("token_export", DEFAULT_TOKEN_EXPORT)
    return argparse.Namespace(**values)

def _load_quantized_layers_json(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or not str(path).strip():
        return []
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"quantized layers JSON must contain a sequence: {target}")
    return [dict(item) for item in raw]


def _pooling_name(value: str) -> str:
    text = str(value).strip()
    if text in {"mask_aware_mean", DEFAULT_POOLING}:
        return DEFAULT_POOLING
    raise ValueError(f"export_eval only supports pooling='mask_aware_mean', got {value!r}")


def _public_export_manifest(
    *,
    config: ExportEvalConfig,
    split_result: Mapping[str, Any],
    token_layer_indices: Sequence[int],
    label_buckets: Mapping[int, str],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(config.checkpoint_path)
    split_csv = Path(config.split_csv)
    label_mapping_path = Path(config.label_mapping_path)
    audio_cfg = audio_config_from_project()
    split_hashes = {str(config.split_name): file_sha256(split_csv)}
    checkpoint_layers = checkpoint.get("quantized_layers") if isinstance(checkpoint.get("quantized_layers"), Sequence) else None
    layers = [dict(item) for item in (checkpoint_layers or config.quantized_layers or [])]
    bucket_hash_payload = {"label_buckets": {str(key): str(value) for key, value in sorted(label_buckets.items())}}
    artifact_sha256 = {"checkpoint": file_sha256(checkpoint_path), **dict(split_result.get("artifact_sha256", {}))}
    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "stage": "baseline_export_eval",
        "model_name": str(config.model_name),
        "model_kind": checkpoint.get("_baseline_model_kind") or _infer_checkpoint_model_kind(checkpoint),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_identity_hash": split_result.get("run_identity_hash"),
        "checkpoint": str(checkpoint_path.resolve(strict=False)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split_name": str(config.split_name),
        "split_csv": {str(config.split_name): str(split_csv.resolve(strict=False))},
        "split_hashes": split_hashes,
        "split_hash": canonical_json_hash(split_hashes),
        "label_mapping": str(label_mapping_path.resolve(strict=False)),
        "label_mapping_hash": file_sha256(label_mapping_path) or checkpoint.get("label_mapping_hash"),
        "bucket_protocol": group_protocol_metadata(),
        "bucket_protocol_source": str(config.bucket_protocol),
        "bucket_protocol_hash": canonical_json_hash(bucket_hash_payload),
        "calibration_hash": config.calibration_hash or checkpoint.get("calibration_hash"),
        "quantized_layers": layers,
        "quantized_layers_hash": checkpoint.get("quantized_layers_hash") or canonical_json_hash({"quantized_layers": layers}),
        "bitwidth": checkpoint.get("bitwidth", config.bitwidth),
        "qparams_hash": checkpoint.get("qparams_hash"),
        "pooling": _pooling_name(config.pooling),
        "pooling_definition": POOLING_DEFINITION,
        "reference_artifact_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_ready_artifacts": (checkpoint.get("_baseline_model_kind") or _infer_checkpoint_model_kind(checkpoint)) == REFERENCE_MODEL_NAME,
        "reference_required_fields": ["logits", "embeddings", "sample_ids", "labels", "sample_order_hash", "label_mapping_hash", "split_hash", "pooling"],
        "oer_protocol": split_result.get("oer_protocol"),
        "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
        "oer_hook_list_used": (split_result.get("oer_protocol") or {}).get("oer_hook_list_used"),
        "oer_hook_list_used_hash": (split_result.get("oer_protocol") or {}).get("oer_hook_list_used_hash"),
        "audio_input_py": str((PROJECT_ROOT / "src" / "data" / "audio_input.py").resolve(strict=False)),
        "audio_input_py_sha256": file_sha256(PROJECT_ROOT / "src" / "data" / "audio_input.py"),
        "input_length_samples": int(audio_cfg.target_num_samples),
        "input_length_seconds": float(audio_cfg.target_seconds),
        "sample_rate": int(audio_cfg.sample_rate),
        "seed": int(config.seed),
        "git_commit": git_commit(),
        "token_export": split_result.get("token_export"),
        "token_layers": [int(item) for item in token_layer_indices],
        "export_all_token_layers": split_result.get("token_export") == "all",
        "deterministic_audio_mode": str(config.split_name),
        "shuffle": False,
        "amp": False,
        "grad_policy": "hard_fail_if_any_param_grad_is_not_None",
        "quant_audit": split_result.get("quant_audit"),
        "outputs": dict(split_result.get("paths", {})),
        "artifact_sha256": artifact_sha256,
        "token_feature_protocol": split_result.get("token_feature_protocol"),
        "metrics_protocol": {
            "source": "src.baseline.baseline_metrics.compute_baseline_metrics",
            "canonical_table_schema": ["model_kind", "split", "seed", "metric_name", "analysis_group", "metric_value", "metric_direction", "aggregation_unit", "aggregation_fn", "metric_role", "support_count", "effective_class_count", "reference_checkpoint_hash", "run_identity_hash", "label_mapping_hash", "split_hash", "sample_order_hash", "pooling"],
            "prediction_source": "logits.argmax(dim=1); external predictions are audit-only and must match",
            "selection_macro_f1_metric": "macro_f1_label_universe_nanmean",
            "macro_f1_observed_classes": "nanmean F1 over support>0 classes in the selected split/group",
            "macro_f1_label_universe_nanmean": "complete label mapping universe; support=0 class F1 is NaN and aggregation is nanmean",
            "macro_f1_legacy_zero_absent_diagnostic": "diagnostic only; support=0 absent classes use zero, never used for selection",
            "per_class_metrics_enumerate_full_label_mapping": True,
            "group_protocol": group_protocol_metadata(),
            "fisher_definition": "Tr(S_B)/(Tr(S_W)+eps)",
            "fisher_normalize_by_dof": False,
            "geo_margin_main": "min_{c != y} ||z - mu_c||_2 - ||z - mu_y||_2",
            "correct_logit_margin_detail": "logit_y - max_{j != y} logit_j",
            "top1_top2_gap_detail": "top1_logit - top2_logit",
            "class_trace_variance": "macro_mean_c Tr(Cov({z_i | y_i = c})); singleton classes are NaN",
            "main_metrics_do_not_depend_on_token_features": True,
            "reference_metrics_require_fp32_reference_bundle": True,
            "embedding_cosine_similarity": "mean cosine_similarity(E_current, E_reference) over pooled embeddings",
            "oer_retention": "sample-level nanmean over H_QI of E_current/E_fp32 using Baseline FP32 teacher OER oracle; target value is 1",
            "oer_abs_deviation_from_1": "abs(oer_retention - 1); lower is better",
            "oer_valid_rate": "fraction of H_QI hooks with valid FP32 oracle energy and non-empty legal oracle positions",
            "oer_hook_point": "BEATs Transformer FFN fc1/fc2 input activation, canonical BTC layout, padding tokens excluded by token-aligned BEATs padding mask",
        },
        "splits": {str(config.split_name): dict(split_result)},
    }
    return attach_run_manifest_hash(manifest)


def run_export_eval_config(config: ExportEvalConfig) -> dict[str, Any]:
    split_name = str(config.split_name).strip().lower()
    if split_name not in {"eval", "test", "export"}:
        raise ValueError(f"split_name must be one of eval/test/export, got {config.split_name!r}")
    pooling = _pooling_name(config.pooling)
    if pooling != DEFAULT_POOLING:
        raise AssertionError("unreachable pooling normalization failure")

    device = torch.device(config.device or getattr(project_config, "DEVICE", "cpu"))
    checkpoint_path = Path(config.checkpoint_path)
    label_mapping_path = Path(config.label_mapping_path)
    split_csv = Path(config.split_csv)
    label_mapping = _load_label_mapping_json(label_mapping_path)
    label_buckets = _resolve_bucket_protocol(config.bucket_protocol, split_csv=split_csv)
    missing_buckets = sorted(set(int(value) for value in label_mapping.values()) - set(int(key) for key in label_buckets))
    if missing_buckets:
        raise ValueError(f"bucket_protocol missing label_id values required by label_mapping: {missing_buckets[:8]}")

    model, checkpoint = load_trained_checkpoint(
        checkpoint_path,
        device=device,
        qparams_csv_override=None if config.qparams_csv_override is None else Path(config.qparams_csv_override),
    )
    _validate_checkpoint_label_mapping(checkpoint, label_mapping)
    args = _export_arg_namespace(config)
    token_layer_indices = _resolve_export_token_layers(model, args)
    token_export = _token_export_mode_from_layers(model, token_layer_indices)
    split_hashes = _split_hash_payload(split_csv)
    run_identity_hash = build_run_identity(
        model_name=str(config.model_name),
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        split_hashes=split_hashes,
        label_mapping_hash=file_sha256(label_mapping_path) or checkpoint.get("label_mapping_hash"),
        calibration_hash=config.calibration_hash or checkpoint.get("calibration_hash"),
        pooling=pooling,
        token_export=token_export,
        token_layers=token_layer_indices,
        seed=int(config.seed),
    )
    args = _formal_export_args(args, run_identity_hash=run_identity_hash, token_export=token_export)
    paths = _SingleSplitPaths(
        name=str(config.model_name),
        root=None if config.dataset_root is None else Path(config.dataset_root),
        train_csv=split_csv,
        eval_csv=split_csv,
        test_csv=split_csv,
        export_csv=split_csv,
        label_mapping=label_mapping_path,
        output_checkpoint=checkpoint_path,
    )
    output_dirs = _flat_export_dirs(config.output_dir)
    split_result = export_eval_split(
        model=model,
        paths=paths,
        split=split_name,
        args=args,
        label_buckets=label_buckets,
        token_layer_indices=token_layer_indices,
        output_dirs=output_dirs,
        device=device,
        checkpoint_path=checkpoint_path,
        quant_controller=getattr(model, "_baseline_quant_controller", None),
        quant_audit=getattr(model, "_baseline_quant_audit", None),
        run_identity_hash=run_identity_hash,
    )
    manifest = _public_export_manifest(
        config=ExportEvalConfig(**{**config.__dict__, "split_name": split_name, "pooling": pooling, "token_export": token_export}),
        split_result=split_result,
        token_layer_indices=token_layer_indices,
        label_buckets=label_buckets,
        checkpoint=checkpoint,
    )
    manifest_path = write_run_manifest(config.output_dir, manifest)
    split_result.setdefault("paths", {})["manifest"] = _path_str(manifest_path)
    return {
        "run_identity_hash": manifest["run_identity_hash"],
        "run_manifest_hash": manifest["run_manifest_hash"],
        "manifest_path": _path_str(manifest_path),
        "split": split_result,
    }


def run_export_eval_once(
    *,
    model_name: str,
    checkpoint_path: str | Path,
    split_csv: str | Path,
    split_name: str,
    output_dir: str | Path,
    label_mapping_path: str | Path,
    bucket_protocol: Any,
    pooling: str = "mask_aware_mean",
    token_layers: Sequence[int] | str = DEFAULT_TOKEN_LAYERS,
    device: str | torch.device | None = None,
    seed: int = 2026,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_export_eval_config(
        ExportEvalConfig(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            split_csv=split_csv,
            split_name=split_name,
            output_dir=output_dir,
            label_mapping_path=label_mapping_path,
            bucket_protocol=bucket_protocol,
            pooling=pooling,
            token_layers=token_layers,
            device=device,
            seed=int(seed),
            **kwargs,
        )
    )

def run_export_eval(
    *,
    paths: Any,
    args: argparse.Namespace,
    label_buckets: Mapping[int, str],
    device: torch.device,
    best_epoch: int,
    best_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    model, checkpoint = load_trained_checkpoint(paths.output_checkpoint, device=device)
    token_layer_indices = _resolve_export_token_layers(model, args)
    token_export = _token_export_mode_from_layers(model, token_layer_indices)
    split_hashes = _split_hash_payload(paths=paths)
    run_identity_hash = build_run_identity(
        model_name=str(args.export_name),
        checkpoint_path=paths.output_checkpoint,
        checkpoint=checkpoint,
        split_hashes=split_hashes,
        label_mapping_hash=file_sha256(paths.label_mapping) or checkpoint.get("label_mapping_hash"),
        calibration_hash=None,
        pooling=DEFAULT_POOLING,
        token_export=token_export,
        token_layers=token_layer_indices,
        seed=int(args.seed),
    )
    export_args = _formal_export_args(args, run_identity_hash=run_identity_hash, token_export=token_export)
    output_dirs = _export_dirs(args.export_root, args.export_name)
    print(
        "[train-head] export_eval "
        f"checkpoint={paths.output_checkpoint} "
        f"splits=eval,test token_layers={list(token_layer_indices)}"
    )
    split_results = {
        split: export_eval_split(
            model=model,
            paths=paths,
            split=split,
            args=export_args,
            label_buckets=label_buckets,
            token_layer_indices=token_layer_indices,
            output_dirs=output_dirs,
            device=device,
            checkpoint_path=paths.output_checkpoint,
            quant_audit=getattr(model, "_baseline_quant_audit", None),
            run_identity_hash=run_identity_hash,
        )
        for split in ("eval", "test")
    }
    audio_cfg = audio_config_from_project()
    bucket_hash_payload = {"label_buckets": {str(key): str(value) for key, value in sorted(label_buckets.items())}}
    artifact_sha256 = {
        "checkpoint": file_sha256(paths.output_checkpoint),
        **{key: value for split_result in split_results.values() for key, value in dict(split_result.get("artifact_sha256", {})).items()},
    }
    outputs_by_split = {str(split): dict(result.get("paths", {})) for split, result in split_results.items()}
    token_feature_protocol_by_split = {
        str(split): result.get("token_feature_protocol") for split, result in split_results.items()
    }
    oer_protocol_by_split = {
        str(split): result.get("oer_protocol") for split, result in split_results.items()
    }
    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "stage": "baseline_export_eval",
        "model_name": str(args.export_name),
        "model_kind": checkpoint.get("_baseline_model_kind") or _infer_checkpoint_model_kind(checkpoint),
        "dataset_name": paths.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_identity_hash": run_identity_hash,
        "checkpoint": str(paths.output_checkpoint.resolve(strict=False)),
        "checkpoint_sha256": file_sha256(paths.output_checkpoint),
        "source_checkpoint": checkpoint.get("source_checkpoint"),
        "split_name": "eval,test",
        "split_csv": {
            "eval": str(paths.eval_csv.resolve(strict=False)),
            "test": str(paths.test_csv.resolve(strict=False)),
        },
        "split_hashes": split_hashes,
        "split_hash": canonical_json_hash(split_hashes),
        "label_mapping": str(paths.label_mapping.resolve(strict=False)),
        "label_mapping_hash": file_sha256(paths.label_mapping) or checkpoint.get("label_mapping_hash"),
        "bucket_protocol": group_protocol_metadata(),
        "bucket_protocol_source": "label_buckets argument from training split metadata",
        "bucket_protocol_hash": canonical_json_hash(bucket_hash_payload),
        "calibration_hash": None,
        "quantized_layers": [],
        "quantized_layers_hash": canonical_json_hash({"quantized_layers": []}),
        "bitwidth": "fp32",
        "qparams_hash": checkpoint.get("qparams_hash"),
        "pooling": DEFAULT_POOLING,
        "pooling_definition": POOLING_DEFINITION,
        "reference_artifact_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_ready_artifacts": (checkpoint.get("_baseline_model_kind") or _infer_checkpoint_model_kind(checkpoint)) == REFERENCE_MODEL_NAME,
        "reference_required_fields": ["logits", "embeddings", "sample_ids", "labels", "sample_order_hash", "label_mapping_hash", "split_hash", "pooling"],
        "oer_protocol": oer_protocol_by_split,
        "oer_activation_position_used": OER_ACTIVATION_POSITION_USED,
        "oer_hook_list_used": _expected_oer_hook_list(),
        "oer_hook_list_used_hash": _oer_hook_list_hash(_expected_oer_hook_list()),
        "audio_input_py": str((PROJECT_ROOT / "src" / "data" / "audio_input.py").resolve(strict=False)),
        "audio_input_py_sha256": file_sha256(PROJECT_ROOT / "src" / "data" / "audio_input.py"),
        "input_length_samples": int(audio_cfg.target_num_samples),
        "input_length_seconds": float(audio_cfg.target_seconds),
        "sample_rate": int(audio_cfg.sample_rate),
        "seed": int(args.seed),
        "git_commit": git_commit(),
        "token_export": token_export,
        "token_layers": [int(item) for item in token_layer_indices],
        "export_all_token_layers": token_export == "all",
        "deterministic_audio_mode": "eval,test",
        "shuffle": False,
        "amp": False,
        "grad_policy": "hard_fail_if_any_param_grad_is_not_None",
        "quant_audit": getattr(model, "_baseline_quant_audit", None),
        "outputs": outputs_by_split,
        "artifact_sha256": artifact_sha256,
        "token_feature_protocol": token_feature_protocol_by_split,
        "metrics_protocol": {
            "source": "src.baseline.baseline_metrics.compute_baseline_metrics",
            "canonical_table_schema": ["model_kind", "split", "seed", "metric_name", "analysis_group", "metric_value", "metric_direction", "aggregation_unit", "aggregation_fn", "metric_role", "support_count", "effective_class_count", "reference_checkpoint_hash", "run_identity_hash", "label_mapping_hash", "split_hash", "sample_order_hash", "pooling"],
            "prediction_source": "logits.argmax(dim=1); external predictions are audit-only and must match",
            "selection_macro_f1_metric": "macro_f1_label_universe_nanmean",
            "macro_f1_observed_classes": "nanmean F1 over support>0 classes in the selected split/group",
            "macro_f1_label_universe_nanmean": "complete label mapping universe; support=0 class F1 is NaN and aggregation is nanmean",
            "macro_f1_legacy_zero_absent_diagnostic": "diagnostic only; support=0 absent classes use zero, never used for selection",
            "per_class_metrics_enumerate_full_label_mapping": True,
            "group_protocol": group_protocol_metadata(),
            "fisher_definition": "Tr(S_B)/(Tr(S_W)+eps)",
            "fisher_normalize_by_dof": False,
            "geo_margin_main": "min_{c != y} ||z - mu_c||_2 - ||z - mu_y||_2",
            "correct_logit_margin_detail": "logit_y - max_{j != y} logit_j",
            "top1_top2_gap_detail": "top1_logit - top2_logit",
            "class_trace_variance": "macro_mean_c Tr(Cov({z_i | y_i = c})); singleton classes are NaN",
            "class_trace_variance_normalization": "dof",
            "main_metrics_do_not_depend_on_token_features": True,
            "reference_metrics_require_fp32_reference_bundle": True,
            "embedding_cosine_similarity": "mean cosine_similarity(E_current, E_reference) over pooled embeddings",
            "oer_retention": "sample-level nanmean over H_QI of E_current/E_fp32 using Baseline FP32 teacher OER oracle; target value is 1",
            "oer_abs_deviation_from_1": "abs(oer_retention - 1); lower is better",
            "oer_valid_rate": "fraction of H_QI hooks with valid FP32 oracle energy and non-empty legal oracle positions",
            "oer_hook_point": "BEATs Transformer FFN fc1/fc2 input activation, canonical BTC layout, padding tokens excluded by token-aligned BEATs padding mask",
            "test_usage_discipline": "test is exported once for final reporting only; no checkpoint, hyperparameter, qparams, temperature, or alpha selection",
        },
        "selection_metric": str(args.selection_metric),
        "best_epoch": int(best_epoch),
        "best_eval_loss": float(best_metrics.get("loss", float("nan"))),
        "best_eval_accuracy": float(best_metrics.get("accuracy", float("nan"))),
        "best_eval_macro_f1_label_universe_nanmean": float(best_metrics.get("eval_macro_f1_label_universe_nanmean", float("nan"))),
        "best_eval_macro_f1_observed_classes": float(best_metrics.get("eval_macro_f1_observed_classes", float("nan"))),
        "best_eval_macro_f1_legacy_zero_absent_diagnostic": float(best_metrics.get("eval_macro_f1_legacy_zero_absent_diagnostic", float("nan"))),
        "export_name": str(args.export_name),
        "export_root": str((Path(args.export_root) if args.export_root else Path(project_config.BASE_DIR) / "outputs").resolve(strict=False)),
        "splits": {str(split): dict(result) for split, result in split_results.items()},
    }
    payload = attach_run_manifest_hash(manifest)
    manifest_paths = write_run_manifests(output_dirs, payload)
    print(f"[train-head] export_eval run_manifest_hash={payload['run_manifest_hash']}")
    print(f"[train-head] export_eval manifest={manifest_paths.get('manifests')}")
    return {
        "run_manifest_hash": payload["run_manifest_hash"],
        "manifest_paths": manifest_paths,
        "splits": split_results,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline export eval: deterministic forward export plus pure baseline metrics.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--split-name", choices=("eval", "test", "export"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-mapping-path", required=True)
    parser.add_argument("--bucket-protocol", required=True, help="JSON/CSV path, or 'split_csv' when split CSV contains canonical bucket metadata.")
    parser.add_argument("--pooling", default="mask_aware_mean")
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--export-all-token-layers", action="store_true", help="Compatibility alias for --token-export all.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-export-payload-gib", type=float, default=DEFAULT_MAX_EXPORT_PAYLOAD_GIB)
    parser.add_argument("--calibration-hash", default=None)
    parser.add_argument("--quantized-layers-json", default=None)
    parser.add_argument("--bitwidth", default="fp32")
    parser.add_argument("--reference-root", default=None, help="Seed export root containing fp32_teacher reference artifacts; required for INT4/KD single-split exports when it cannot be inferred.")
    parser.add_argument("--qparams-csv-override", default=None, help="Current-machine qparams CSV to use when an INT4/KD checkpoint was moved across filesystems.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    summary = run_export_eval_config(
        ExportEvalConfig(
            model_name=str(args.model_name),
            checkpoint_path=Path(args.checkpoint_path),
            split_csv=Path(args.split_csv),
            split_name=str(args.split_name),
            output_dir=Path(args.output_dir),
            label_mapping_path=Path(args.label_mapping_path),
            bucket_protocol=str(args.bucket_protocol),
            pooling=str(args.pooling),
            token_export=str(args.token_export),
            token_layers=str(args.token_layers),
            device=args.device,
            seed=int(args.seed),
            dataset_root=None if args.dataset_root is None else Path(args.dataset_root),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            amp=False,
            export_all_token_layers=bool(args.export_all_token_layers),
            max_export_payload_gib=float(args.max_export_payload_gib),
            calibration_hash=args.calibration_hash,
            quantized_layers=_load_quantized_layers_json(args.quantized_layers_json),
            bitwidth=str(args.bitwidth),
            reference_root=None if getattr(args, "reference_root", None) is None else Path(args.reference_root),
            qparams_csv_override=None if getattr(args, "qparams_csv_override", None) is None else Path(args.qparams_csv_override),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()















