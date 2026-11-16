from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
import src.export.single_layer_causal_injection as single_layer
import src.export.three_factor_decomposition as three_factor
from src.baseline.export_eval import pooled_logits_from_features, prepare_encoder_input_for_export
from src.quant.quant_linear import QuantLinear
from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
from src.utils.semantic_geometry_cache import file_identity, qparams_identity_manifest, stable_json_hash


EXPERIMENT_NAME = "decision_harmful_token_ranking"
PROTOCOL_VERSION = "decision_harmful_token_ranking.v1"
DEFAULT_OUTPUT_SUBDIR = EXPERIMENT_NAME
DEFAULT_OER_ORACLE_RELATIVE = (
    Path("cache")
    / "fp32_reference"
    / "oer_oracle_eval.pt"
)
THREE_FACTOR_SAMPLE_FILENAME = "three_factor_sample.csv"
DETAIL_FILENAME = "decision_harmful_token_detail.csv"
SUMMARY_FILENAME = "decision_harmful_token_summary.csv"
GROUP_SUMMARY_FILENAME = "decision_harmful_token_group_summary.csv"
MANIFEST_FILENAME = "manifest.json"
REPRESENTATION_HOOK_POSITION = "linear_module_output"
OER_HOOK_POSITION = "ffn_input_activation"
EPS = 1e-12
DEFAULT_NUMERICAL_TOLERANCE = 1e-6
TOPK_RATIO_TOLERANCE = 1e-9
GROUPS = ("overall", "many", "medium", "tail", "few")
FLIP_STATUSES = ("all", "nonflip", "flip")


DETAIL_COLUMNS = [
    "sample_index",
    "sample_id",
    "label_id",
    "bucket",
    "target",
    "token_index",
    "valid_token",
    "fp32_energy",
    "perturbation_norm",
    "decision_projection_C",
    "harmful_score",
    "oer_count",
    "oer_abs_error",
    "rank_harmful",
    "rank_energy",
    "rank_distortion",
    "rank_oer",
    "is_topk_harmful",
    "is_topk_energy",
    "is_topk_distortion",
    "is_topk_oer",
]


SUMMARY_COLUMNS = [
    "sample_index",
    "sample_id",
    "label_id",
    "bucket",
    "target",
    "fp32_pred",
    "injected_pred",
    "j_star",
    "m_fp32",
    "m_injected",
    "classification_flip",
    "valid_token_count",
    "K_eff",
    "harmful_energy_overlap",
    "harmful_distortion_overlap",
    "harmful_oer_overlap",
    "harmful_energy_spearman",
    "harmful_distortion_spearman",
    "harmful_oer_spearman",
    "sum_decision_projection",
    "first_order_margin_change",
    "predicted_margin_drop",
    "actual_margin_drop",
    "three_factor_first_order",
    "three_factor_actual_margin_drop",
    "three_factor_reference_abs_error",
    "three_factor_csv_abs_error",
    "three_factor_csv_within_tolerance",
    "three_factor_actual_margin_drop_abs_error",
    "three_factor_actual_margin_drop_within_tolerance",
    "three_factor_classification_fields_match",
    "oer_scalar_k",
    "oer_topk_ratio",
]


GROUP_SUMMARY_COLUMNS = [
    "target",
    "group",
    "flip_status",
    "n",
    "mean_harmful_energy_overlap",
    "median_harmful_energy_overlap",
    "std_harmful_energy_overlap",
    "mean_harmful_distortion_overlap",
    "median_harmful_distortion_overlap",
    "std_harmful_distortion_overlap",
    "mean_harmful_oer_overlap",
    "median_harmful_oer_overlap",
    "std_harmful_oer_overlap",
    "mean_harmful_energy_spearman",
    "median_harmful_energy_spearman",
    "std_harmful_energy_spearman",
    "mean_harmful_distortion_spearman",
    "median_harmful_distortion_spearman",
    "std_harmful_distortion_spearman",
    "mean_harmful_oer_spearman",
    "median_harmful_oer_spearman",
    "std_harmful_oer_spearman",
    "flip_count",
    "flip_rate",
]


@dataclass(frozen=True)
class OerOracleRow:
    sample_id: str
    target: str
    layer_idx: int
    module_name: str
    activation_position: str
    hook_name: str
    scalar_indices: Tensor
    topk_k: int
    topk_ratio: float
    valid_token_count: int
    valid_scalar_count: int
    feature_dim: int
    source_shard: str


@dataclass(frozen=True)
class OerOracleIndex:
    path: Path
    manifest_path: Path | None
    payload_manifest: Mapping[str, Any]
    rows_by_key: Mapping[tuple[str, str], OerOracleRow]
    sample_id_order: tuple[str, ...]
    shard_paths: tuple[str, ...]
    topk_ratios: tuple[float, ...]
    protocol_version: str


@dataclass(frozen=True)
class ThreeFactorReferenceRow:
    sample_id: str
    target: str
    first_order_margin_change: float
    actual_margin_drop: float
    classification_flip: int
    j_star: int
    fp32_pred: int
    injected_pred: int


@dataclass(frozen=True)
class TargetTokenRankingResult:
    spec: single_layer.TargetSpec
    sample_order: tuple[str, ...]
    fp32_logits: Tensor
    injected_logits: Tensor
    fp32_preds: Tensor
    injected_preds: Tensor
    detail_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    quant_state: tuple[dict[str, Any], ...]
    single_layer_reference_validation: Mapping[str, Any]
    checkpoint_clean_reloads: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
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
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = dict(single_layer._resolve_paths(args))
    output_dir = Path(args.output_dir) if args.output_dir is not None else paths["dataset_root"] / DEFAULT_OUTPUT_SUBDIR
    oer_oracle = Path(args.oer_oracle) if args.oer_oracle is not None else paths["dataset_root"] / DEFAULT_OER_ORACLE_RELATIVE
    three_factor_csv = (
        Path(args.three_factor_sample_csv)
        if args.three_factor_sample_csv is not None
        else paths["dataset_root"] / three_factor.DEFAULT_OUTPUT_SUBDIR / THREE_FACTOR_SAMPLE_FILENAME
    )
    if not oer_oracle.exists():
        raise FileNotFoundError(
            f"fixed OER oracle not found: {oer_oracle}. "
            "This script refuses to rebuild or reselect the oracle."
        )
    if not three_factor_csv.exists():
        raise FileNotFoundError(
            f"three-factor reference CSV not found: {three_factor_csv}. "
            "Run src.export.three_factor_decomposition first, or pass --three-factor-sample-csv."
        )
    manifest_candidates = (
        oer_oracle.with_suffix(oer_oracle.suffix + ".manifest.json"),
        oer_oracle.with_name("fp32_oer_oracle.manifest.json"),
        oer_oracle.with_suffix(".manifest.json"),
    )
    manifest_candidate = next((candidate for candidate in manifest_candidates if candidate.exists()), manifest_candidates[0])
    paths["output_dir"] = output_dir
    paths["oer_oracle"] = oer_oracle
    paths["oer_oracle_manifest"] = manifest_candidate
    paths["three_factor_sample_csv"] = three_factor_csv
    return paths


def _validate_args(args: argparse.Namespace) -> None:
    if str(args.dataset) != single_layer.DEFAULT_DATASET:
        raise ValueError(f"{EXPERIMENT_NAME} only accepts dataset={single_layer.DEFAULT_DATASET!r}; got {args.dataset!r}")
    if int(args.batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size!r}")
    if int(args.num_workers) < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers!r}")
    if float(args.numerical_tolerance) <= 0.0:
        raise ValueError(f"numerical_tolerance must be positive, got {args.numerical_tolerance!r}")
    for forbidden in ("force", "rebuild", "recalibrate", "train", "use_test"):
        if bool(getattr(args, forbidden, False)):
            raise ValueError(f"{EXPERIMENT_NAME} refuses forbidden option --{forbidden.replace('_', '-')}")


def _assert_center_crop_protocol(batch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    audio_cfg = single_layer._cfg_get(project_config, "AUDIO", {})
    configured = audio_cfg.get("CROP_MODE") if isinstance(audio_cfg, Mapping) else None
    if str(configured).strip().lower() != "center":
        raise RuntimeError(f"AUDIO.CROP_MODE must remain 'center'; got {configured!r}")
    if isinstance(batch, Mapping):
        for key in ("crop_mode", "audio_crop_mode", "CROP_MODE"):
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, Tensor):
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values = {str(item).strip().lower() for item in value}
            else:
                values = {str(value).strip().lower()}
            bad = sorted(item for item in values if item and item != "center")
            if bad:
                raise RuntimeError(f"eval batch crop protocol must remain center crop; {key}={bad[:4]}")
    return {"crop_mode": "center", "source": "config.AUDIO.CROP_MODE"}


def _canonicalize(value: Tensor, token_padding_mask: Tensor, *, context: str) -> tuple[Tensor, dict[str, Any]]:
    canonical = canonicalize_activation_btc(
        value,
        padding_mask=token_padding_mask,
        batch_size=int(token_padding_mask.shape[0]),
        token_count=int(token_padding_mask.shape[1]),
        context=context,
    )
    return canonical.tensor, layout_info_to_public_fields(canonical.layout_info)


def _canonicalize_like(value: Tensor, token_padding_mask: Tensor, layout_info: Mapping[str, Any], *, context: str) -> Tensor:
    raw_layout = str(layout_info.get("original_layout") or "")
    if not raw_layout:
        raise RuntimeError(f"{context}: missing original_layout")
    canonical = canonicalize_activation_btc(
        value,
        raw_layout=raw_layout,
        padding_mask=token_padding_mask,
        batch_size=int(token_padding_mask.shape[0]),
        token_count=int(token_padding_mask.shape[1]),
        context=context,
    )
    return canonical.tensor


def _capture_for_grad(value: Tensor) -> Tensor:
    if value.requires_grad:
        return value
    return value.detach().requires_grad_(True)


def _assert_equal_tensor(name: str, left: Tensor, right: Tensor) -> None:
    if torch.equal(left, right):
        return
    diff = torch.max(torch.abs(left.detach().cpu().float() - right.detach().cpu().float())).item()
    raise RuntimeError(f"{name}: tensors are not exactly identical; max_abs_diff={float(diff):.8g}")


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _read_json_if_exists(path: Path | None) -> Mapping[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, Mapping) else {}


def _resolve_shard_path(oracle_path: Path, entry: Mapping[str, Any]) -> Path:
    raw = entry.get("path", entry.get("file_path", entry.get("shard_path")))
    if raw is None:
        raise KeyError("OER oracle shard entry is missing path/file_path/shard_path")
    text = str(raw).strip()
    path = Path(text)
    if path.is_absolute():
        return path
    return oracle_path.parent / path


def _column_item(value: Any, row_idx: int, row_count: int) -> Any:
    if isinstance(value, Tensor):
        if value.ndim >= 1 and int(value.shape[0]) == int(row_count):
            return value[row_idx]
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == int(row_count):
            return value[row_idx]
    return value


def _iter_oracle_rows_from_payload(payload: Mapping[str, Any], *, source_shard: str) -> list[Mapping[str, Any]]:
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [dict(row, source_shard=source_shard) for row in rows if isinstance(row, Mapping)]

    sample_ids = payload.get("sample_id")
    if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)):
        result: list[Mapping[str, Any]] = []
        row_count = len(sample_ids)
        for row_idx in range(row_count):
            row: dict[str, Any] = {"source_shard": source_shard}
            for key, value in payload.items():
                if key in {"manifest", "rows", "shards", "row_index", "sample_id_set"}:
                    continue
                row[key] = _column_item(value, row_idx, row_count)
            result.append(row)
        return result
    return []


def _row_target(row: Mapping[str, Any]) -> str:
    return f"L{int(row.get('layer_idx'))}.{str(row.get('module_name'))}"


def _as_scalar_indices(row: Mapping[str, Any]) -> Tensor:
    for key in ("topk_scalar_indices", "input_topk_indices", "topk_indices", "flat_index"):
        if key in row and row[key] is not None:
            tensor = torch.as_tensor(row[key], dtype=torch.long).detach().cpu().reshape(-1).contiguous()
            if int(tensor.numel()) <= 0:
                raise ValueError(f"OER oracle row has empty scalar indices: {row.get('sample_id')} {_row_target(row)}")
            return tensor
    raise KeyError(f"OER oracle row missing scalar top-k indices: keys={sorted(row.keys())}")


def _infer_feature_dim(row: Mapping[str, Any]) -> int:
    value = row.get("feature_dim")
    if value is not None:
        return int(torch.as_tensor(value).reshape(()).item())
    for key in ("activation_shape", "canonical_shape", "original_shape"):
        shape = row.get(key)
        if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and shape:
            return int(shape[-1])
    raise KeyError(f"OER oracle row missing feature_dim: {row.get('sample_id')} {_row_target(row)}")


def _manifest_value(payload: Mapping[str, Any], manifest: Mapping[str, Any], key: str) -> Any:
    return manifest[key] if key in manifest else payload.get(key)


def _require_manifest_value(payload: Mapping[str, Any], manifest: Mapping[str, Any], key: str) -> Any:
    value = _manifest_value(payload, manifest, key)
    if value is None:
        raise RuntimeError(f"fixed OER oracle manifest is missing required field {key!r}")
    return value


def _assert_oer_oracle_protocol(path: Path, payload: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    cache_type = str(_require_manifest_value(payload, manifest, "cache_type"))
    if cache_type != "fp32_oer_oracle":
        raise RuntimeError(f"{path}: expected cache_type='fp32_oer_oracle', got {cache_type!r}")
    storage_format = str(_require_manifest_value(payload, manifest, "storage_format"))
    if storage_format != "layer_sample_sharded_v1":
        raise RuntimeError(f"{path}: expected storage_format='layer_sample_sharded_v1', got {storage_format!r}")
    if not bool(_require_manifest_value(payload, manifest, "sharded")):
        raise RuntimeError(f"{path}: fixed OER oracle must be sharded")
    oracle_source = str(_require_manifest_value(payload, manifest, "oracle_source"))
    if oracle_source != "fp32_real_activation":
        raise RuntimeError(f"{path}: expected oracle_source='fp32_real_activation', got {oracle_source!r}")
    topk_unit = str(_require_manifest_value(payload, manifest, "topk_unit"))
    if topk_unit != "scalar":
        raise RuntimeError(f"{path}: expected topk_unit='scalar', got {topk_unit!r}")
    topk_source = str(_require_manifest_value(payload, manifest, "topk_source"))
    if topk_source != "fp32_scalar_energy":
        raise RuntimeError(f"{path}: expected topk_source='fp32_scalar_energy', got {topk_source!r}")
    valid_unit = str(_require_manifest_value(payload, manifest, "valid_unit"))
    if valid_unit != "valid_scalar":
        raise RuntimeError(f"{path}: expected valid_unit='valid_scalar', got {valid_unit!r}")
    if not bool(_require_manifest_value(payload, manifest, "padding_excluded")):
        raise RuntimeError(f"{path}: fixed OER oracle must exclude padding tokens")
    positions = _require_manifest_value(payload, manifest, "activation_positions")
    if isinstance(positions, str):
        position_set = {positions.strip().lower()}
    elif isinstance(positions, Sequence):
        position_set = {str(item).strip().lower() for item in positions}
    else:
        raise RuntimeError(f"{path}: activation_positions must be a sequence, got {type(positions)!r}")
    if "input" not in position_set:
        raise RuntimeError(f"{path}: fixed OER oracle must include FFN input activation rows")

    expected_topk_ratio = float(project_config.OER_SCALAR_TOP_RATIO)
    topk_ratio = float(_require_manifest_value(payload, manifest, "topk_ratio"))
    if not math.isclose(topk_ratio, expected_topk_ratio, rel_tol=0.0, abs_tol=TOPK_RATIO_TOLERANCE):
        raise RuntimeError(f"{path}: expected fixed scalar OER topk_ratio={expected_topk_ratio}, got {topk_ratio}")
    topk_policy = str(_require_manifest_value(payload, manifest, "topk_k_policy"))
    if "ceil" not in topk_policy or "valid" not in topk_policy:
        raise RuntimeError(f"{path}: unexpected topk_k_policy={topk_policy!r}")
    coordinate_system = str(_require_manifest_value(payload, manifest, "scalar_flatten_coordinate_system"))
    if "token_idx" not in coordinate_system or "feature" not in coordinate_system:
        raise RuntimeError(f"{path}: scalar coordinate system does not describe token/channel indices")
    target_modules = _require_manifest_value(payload, manifest, "target_modules")
    if not isinstance(target_modules, Sequence) or isinstance(target_modules, (str, bytes)):
        raise RuntimeError(f"{path}: target_modules must be a sequence")
    available_targets = {
        (int(item["layer_idx"]), str(item["module_name"]))
        for item in target_modules
        if isinstance(item, Mapping) and "layer_idx" in item and "module_name" in item
    }
    for spec in single_layer.TARGET_SPECS:
        if (int(spec.layer_idx), str(spec.module_name)) not in available_targets:
            raise RuntimeError(f"{path}: fixed OER oracle is missing target {spec.target}")
    return {
        "cache_type": cache_type,
        "storage_format": storage_format,
        "oracle_source": oracle_source,
        "topk_unit": topk_unit,
        "topk_source": topk_source,
        "topk_ratio": topk_ratio,
        "padding_excluded": True,
        "activation_positions": sorted(position_set),
    }


def _load_oer_oracle_index(path: Path, manifest_path: Path | None) -> OerOracleIndex:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"OER oracle must be a mapping, got {type(payload)!r}: {path}")
    payload_manifest = payload.get("manifest", {})
    if not isinstance(payload_manifest, Mapping):
        payload_manifest = {}
    external_manifest = _read_json_if_exists(manifest_path)
    manifest = dict(payload_manifest)
    manifest.update({key: value for key, value in external_manifest.items() if key not in manifest})
    _assert_oer_oracle_protocol(path, payload, manifest)

    target_names = {spec.target for spec in single_layer.TARGET_SPECS}
    source_payloads: list[tuple[Mapping[str, Any], str]] = [(payload, str(path))]
    shard_paths: list[str] = []
    shards = payload.get("shards")
    if isinstance(shards, Sequence) and not isinstance(shards, (str, bytes)):
        source_payloads = []
        for entry in shards:
            if not isinstance(entry, Mapping):
                raise TypeError(f"OER oracle shard entry must be a mapping, got {type(entry)!r}")
            shard_path = _resolve_shard_path(path, entry)
            if not shard_path.exists():
                raise FileNotFoundError(f"OER oracle shard missing: {shard_path}")
            shard_payload = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard_payload, Mapping):
                raise TypeError(f"OER oracle shard must be a mapping, got {type(shard_payload)!r}: {shard_path}")
            source_payloads.append((shard_payload, str(shard_path)))
            shard_paths.append(str(shard_path.resolve(strict=False)))

    rows_by_key: dict[tuple[str, str], OerOracleRow] = {}
    topk_ratios: set[float] = set()
    for source, source_shard in source_payloads:
        for raw_row in _iter_oracle_rows_from_payload(source, source_shard=source_shard):
            if "layer_idx" not in raw_row or "module_name" not in raw_row:
                continue
            target = _row_target(raw_row)
            if target not in target_names:
                continue
            activation_position = str(raw_row.get("activation_position", raw_row.get("position", ""))).lower()
            if activation_position != "input":
                continue
            sample_id = str(raw_row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"OER oracle row missing sample_id for {target}")
            scalar_indices = _as_scalar_indices(raw_row)
            topk_k = int(raw_row.get("topk_k", int(scalar_indices.numel())))
            if int(scalar_indices.numel()) != int(topk_k):
                raise RuntimeError(
                    f"{sample_id} {target}: OER scalar index count {int(scalar_indices.numel())} "
                    f"does not match topk_k={topk_k}"
                )
            feature_dim = _infer_feature_dim(raw_row)
            valid_token_count = int(raw_row.get("valid_token_count", -1))
            valid_scalar_count = int(raw_row.get("valid_scalar_count", -1))
            topk_ratio = float(raw_row.get("topk_ratio", manifest.get("topk_ratio", project_config.OER_SCALAR_TOP_RATIO)))
            hook_name = str(raw_row.get("hook_name", f"encoder.layers.{int(raw_row.get('layer_idx'))}.{raw_row.get('module_name')}.input"))
            key = (sample_id, target)
            if key in rows_by_key:
                raise RuntimeError(f"duplicate OER oracle row for {sample_id} {target}")
            rows_by_key[key] = OerOracleRow(
                sample_id=sample_id,
                target=target,
                layer_idx=int(raw_row.get("layer_idx")),
                module_name=str(raw_row.get("module_name")),
                activation_position=activation_position,
                hook_name=hook_name,
                scalar_indices=scalar_indices,
                topk_k=topk_k,
                topk_ratio=topk_ratio,
                valid_token_count=valid_token_count,
                valid_scalar_count=valid_scalar_count,
                feature_dim=feature_dim,
                source_shard=source_shard,
            )
            topk_ratios.add(float(topk_ratio))

    expected_topk_ratio = float(project_config.OER_SCALAR_TOP_RATIO)
    for ratio in topk_ratios:
        if not math.isclose(float(ratio), expected_topk_ratio, rel_tol=0.0, abs_tol=TOPK_RATIO_TOLERANCE):
            raise RuntimeError(
                f"{path}: OER row topk_ratio changed; expected fixed {expected_topk_ratio}, got {ratio}"
            )
    for spec in single_layer.TARGET_SPECS:
        if not any(target == spec.target for _sample_id, target in rows_by_key):
            raise RuntimeError(f"fixed OER oracle contains no rows for {spec.target}")
    sample_order_raw = manifest.get("sample_id_order", payload.get("sample_id_order", payload.get("sample_id", [])))
    sample_order = tuple(str(item) for item in sample_order_raw) if isinstance(sample_order_raw, Sequence) and not isinstance(sample_order_raw, (str, bytes)) else ()
    protocol_raw = manifest.get("protocol_version", payload.get("protocol_version", None))
    if protocol_raw is None or not str(protocol_raw).strip():
        cache_type = str(_manifest_value(payload, manifest, "cache_type") or "fp32_oer_oracle")
        oracle_version = _manifest_value(payload, manifest, "oracle_version") or payload.get("oracle_version", "unknown")
        protocol_version = f"{cache_type}.v{oracle_version}"
    else:
        protocol_version = str(protocol_raw)
    return OerOracleIndex(
        path=path,
        manifest_path=manifest_path if manifest_path is not None and manifest_path.exists() else None,
        payload_manifest=manifest,
        rows_by_key=rows_by_key,
        sample_id_order=sample_order,
        shard_paths=tuple(shard_paths),
        topk_ratios=tuple(sorted(topk_ratios)),
        protocol_version=protocol_version,
    )


def _load_three_factor_reference(path: Path) -> dict[tuple[str, str], ThreeFactorReferenceRow]:
    rows: dict[tuple[str, str], ThreeFactorReferenceRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "target",
            "first_order_margin_change",
            "actual_margin_drop",
            "classification_flip",
            "j_star",
            "fp32_pred",
            "injected_pred",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"three-factor reference CSV missing required columns: {missing}")
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            target = str(row.get("target", "")).strip()
            if not sample_id or not target:
                continue
            key = (sample_id, target)
            if key in rows:
                raise RuntimeError(f"duplicate three-factor reference row for {sample_id} {target}")
            rows[key] = ThreeFactorReferenceRow(
                sample_id=sample_id,
                target=target,
                first_order_margin_change=float(row["first_order_margin_change"]),
                actual_margin_drop=float(row["actual_margin_drop"]),
                classification_flip=int(float(row["classification_flip"])),
                j_star=int(float(row["j_star"])),
                fp32_pred=int(float(row["fp32_pred"])),
                injected_pred=int(float(row["injected_pred"])),
            )
    if not rows:
        raise ValueError(f"three-factor reference CSV contains no rows: {path}")
    return rows


def _forward_fp32_with_hooks(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    spec: single_layer.TargetSpec,
) -> dict[str, Any]:
    model.eval()
    single_layer._set_no_param_grad(model)
    moved = single_layer._move_batch_to_device(batch, device)
    labels = moved["label_id"].long()
    captured: dict[str, Tensor] = {}
    target_module = single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name)

    def pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del module, kwargs
        if not args or not isinstance(args[0], Tensor):
            raise RuntimeError(f"{spec.target}: input hook did not receive a Tensor")
        captured["input_raw"] = args[0].detach()

    def post_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Tensor:
        del module, args, kwargs
        if not isinstance(output, Tensor):
            raise RuntimeError(f"{spec.target}: output hook did not receive a Tensor")
        replacement = _capture_for_grad(output)
        captured["output_raw"] = replacement
        return replacement

    pre_handle = target_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = target_module.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.enable_grad():
            encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
                model,
                moved["waveform"],
                moved["padding_mask"],
            )
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            logits = pooled_logits_from_features(model, features.float(), token_padding_mask)
            if "input_raw" not in captured:
                raise RuntimeError(f"{spec.target}: input hook did not fire")
            if "output_raw" not in captured:
                raise RuntimeError(f"{spec.target}: output hook did not fire")
            x_fp32, input_layout = _canonicalize(
                captured["input_raw"],
                token_padding_mask,
                context=f"{spec.target}.fp32_input",
            )
            h_fp32, output_layout = _canonicalize(
                captured["output_raw"],
                token_padding_mask,
                context=f"{spec.target}.fp32_output",
            )
            preds = logits.detach().argmax(dim=1)
            fp32_correct = preds.eq(labels)
            j_star = single_layer._competitor_indices(logits.detach(), labels)
            m_fp32 = single_layer._fixed_margin(logits, labels, j_star)
            gradient = torch.zeros_like(h_fp32)
            if bool(fp32_correct.any().item()):
                selected_margin = m_fp32[fp32_correct].sum()
                (grad_raw,) = torch.autograd.grad(
                    selected_margin,
                    captured["output_raw"],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                gradient = _canonicalize_like(
                    grad_raw.detach(),
                    token_padding_mask,
                    output_layout,
                    context=f"{spec.target}.fp32_output_gradient",
                )
    finally:
        pre_handle.remove()
        post_handle.remove()

    single_layer._assert_no_param_grads(model, context=f"{spec.target}.fp32_decision_harmful_forward")
    return {
        "logits": logits.detach().cpu().float(),
        "labels": labels.detach().cpu().long(),
        "preds": preds.detach().cpu().long(),
        "j_star": j_star.detach().cpu().long(),
        "m_fp32": m_fp32.detach().cpu().float(),
        "dynamic_margin": single_layer._dynamic_margin(logits.detach(), labels).detach().cpu().float(),
        "fp32_correct": fp32_correct.detach().cpu().bool(),
        "H": h_fp32.detach().cpu().float(),
        "G": gradient.detach().cpu().float(),
        "X_fp32": x_fp32.detach().cpu().float(),
        "token_padding_mask": token_padding_mask.detach().cpu().bool(),
        "input_layout_info": dict(input_layout),
        "output_layout_info": dict(output_layout),
    }


def _quant_target_module(model: nn.Module, spec: single_layer.TargetSpec) -> QuantLinear:
    module = single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name)
    if not isinstance(module, QuantLinear):
        raise TypeError(f"{spec.target}: expected QuantLinear target module, got {type(module)!r}")
    return module


def _forward_injected_with_hooks(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    spec: single_layer.TargetSpec,
) -> dict[str, Any]:
    single_layer._assert_single_target_quant_state(model, spec)
    model.eval()
    single_layer._set_no_param_grad(model)
    moved = single_layer._move_batch_to_device(batch, device)
    labels = moved["label_id"].long()
    captured: dict[str, Tensor] = {}
    target_module = _quant_target_module(model, spec)
    target_module.clear_oer_activation_tensors()

    def post_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del module, args, kwargs
        if not isinstance(output, Tensor):
            raise RuntimeError(f"{spec.target}: injected output hook did not receive a Tensor")
        captured["output_raw"] = output.detach()

    handle = target_module.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
                model,
                moved["waveform"],
                moved["padding_mask"],
            )
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            logits = pooled_logits_from_features(model, features.float(), token_padding_mask)
            if "output_raw" not in captured:
                raise RuntimeError(f"{spec.target}: injected output hook did not fire")
            h_q, output_layout = _canonicalize(
                captured["output_raw"],
                token_padding_mask,
                context=f"{spec.target}.injected_output",
            )
            x_fp32_injected, x_a4 = target_module.get_oer_activation_tensors()
            valid_mask = target_module.last_activation_valid_mask_for_oer
            input_layout = dict(target_module.last_activation_layout_info_for_oer or {})
            if x_fp32_injected is None or x_a4 is None:
                raise RuntimeError(f"{spec.target}: QuantLinear did not expose FP32/A4 input activations")
            if valid_mask is None:
                raise RuntimeError(f"{spec.target}: QuantLinear did not expose valid token mask for OER input")
            x_fp32_injected = x_fp32_injected.detach().cpu().float()
            x_a4 = x_a4.detach().cpu().float()
            valid_mask = valid_mask.detach().cpu().bool()
            expected_valid = ~token_padding_mask.detach().cpu().bool()
            _assert_equal_tensor(f"{spec.target}.injected_input_valid_mask", valid_mask, expected_valid)
            if tuple(x_fp32_injected.shape) != tuple(x_a4.shape):
                raise RuntimeError(
                    f"{spec.target}: injected input FP32/A4 shape mismatch: "
                    f"{tuple(x_fp32_injected.shape)} vs {tuple(x_a4.shape)}"
                )
            if tuple(x_fp32_injected.shape[:2]) != tuple(expected_valid.shape):
                raise RuntimeError(
                    f"{spec.target}: injected input activation shape {tuple(x_fp32_injected.shape)} "
                    f"does not align token mask {tuple(expected_valid.shape)}"
                )
    finally:
        handle.remove()

    target_module.clear_oer_activation_tensors()
    single_layer._assert_single_target_quant_state(model, spec)
    single_layer._assert_no_param_grads(model, context=f"{spec.target}.injected_decision_harmful_forward")
    return {
        "logits": logits.detach().cpu().float(),
        "labels": labels.detach().cpu().long(),
        "preds": logits.detach().argmax(dim=1).cpu().long(),
        "dynamic_margin": single_layer._dynamic_margin(logits.detach(), labels).detach().cpu().float(),
        "H_q": h_q.detach().cpu().float(),
        "X_fp32_injected": x_fp32_injected,
        "X_a4": x_a4,
        "token_padding_mask": token_padding_mask.detach().cpu().bool(),
        "input_layout_info": input_layout,
        "output_layout_info": dict(output_layout),
    }


def _fixed_token_topk_ratio() -> float:
    ratio = float(getattr(project_config, "TOKEN_TOPK_RATIO", 0.05))
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"project_config.TOKEN_TOPK_RATIO must be in (0, 1], got {ratio!r}")
    return ratio


def _fixed_token_k(valid_token_count: int) -> int:
    count = int(valid_token_count)
    if count <= 0:
        raise ValueError("valid_token_count must be positive")
    return max(1, min(count, int(math.ceil(float(count) * _fixed_token_topk_ratio()))))


def _descending_order(scores: Tensor, token_indices: Tensor) -> list[int]:
    values = [float(item) for item in scores.detach().cpu().reshape(-1).tolist()]
    tokens = [int(item) for item in token_indices.detach().cpu().reshape(-1).tolist()]
    return sorted(range(len(values)), key=lambda idx: (-values[idx], tokens[idx]))


def _ordinal_ranks(scores: Tensor, token_indices: Tensor) -> Tensor:
    order = _descending_order(scores, token_indices)
    ranks = torch.empty(int(scores.numel()), dtype=torch.long)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return ranks


def _topk_token_set(scores: Tensor, token_indices: Tensor, k_eff: int) -> set[int]:
    order = _descending_order(scores, token_indices)
    tokens = token_indices.detach().cpu().long().reshape(-1)
    return {int(tokens[idx].item()) for idx in order[: int(k_eff)]}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_scores(left: Tensor, right: Tensor) -> float:
    x = np.asarray(left.detach().cpu().double().reshape(-1).tolist(), dtype=np.float64)
    y = np.asarray(right.detach().cpu().double().reshape(-1).tolist(), dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _overlap(left: set[int], right: set[int], k_eff: int) -> float:
    if int(k_eff) <= 0:
        return float("nan")
    return float(len(left.intersection(right)) / int(k_eff))


def _valid_sample_tensors(
    *,
    h_fp32: Tensor,
    h_q: Tensor,
    gradient: Tensor,
    x_fp32: Tensor,
    x_a4: Tensor,
    token_padding_mask: Tensor,
    sample_index: int,
    spec: single_layer.TargetSpec,
) -> dict[str, Tensor]:
    if tuple(h_fp32.shape) != tuple(h_q.shape) or tuple(h_fp32.shape) != tuple(gradient.shape):
        raise RuntimeError(
            f"{spec.target}: H/H_q/G shape mismatch: "
            f"H={tuple(h_fp32.shape)} H_q={tuple(h_q.shape)} G={tuple(gradient.shape)}"
        )
    if h_fp32.ndim != 3 or x_fp32.ndim != 3 or x_a4.ndim != 3:
        raise RuntimeError(f"{spec.target}: expected rank-3 BTC tensors for output and input activations")
    if tuple(x_fp32.shape) != tuple(x_a4.shape):
        raise RuntimeError(f"{spec.target}: X_fp32/X_A4 shape mismatch: {tuple(x_fp32.shape)} vs {tuple(x_a4.shape)}")
    if tuple(token_padding_mask.shape) != tuple(h_fp32.shape[:2]):
        raise RuntimeError(f"{spec.target}: output token mask mismatch")
    if tuple(token_padding_mask.shape) != tuple(x_fp32.shape[:2]):
        raise RuntimeError(f"{spec.target}: input token mask mismatch")

    valid_mask = ~token_padding_mask[int(sample_index)].bool()
    token_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().cpu().long()
    if int(token_indices.numel()) <= 0:
        raise RuntimeError(f"{spec.target}: sample index {sample_index} has zero valid tokens")
    return {
        "token_indices": token_indices,
        "H": h_fp32[int(sample_index)][valid_mask].detach().cpu().double(),
        "H_q": h_q[int(sample_index)][valid_mask].detach().cpu().double(),
        "G": gradient[int(sample_index)][valid_mask].detach().cpu().double(),
        "X_fp32_full": x_fp32[int(sample_index)].detach().cpu().double(),
        "X_a4_full": x_a4[int(sample_index)].detach().cpu().double(),
        "valid_mask_full": valid_mask.detach().cpu().bool(),
    }


def _aggregate_oer_to_tokens(
    *,
    oracle_row: OerOracleRow,
    x_fp32_full: Tensor,
    x_a4_full: Tensor,
    valid_mask_full: Tensor,
    token_indices: Tensor,
    spec: single_layer.TargetSpec,
    sample_id: str,
) -> tuple[Tensor, Tensor]:
    if tuple(x_fp32_full.shape) != tuple(x_a4_full.shape):
        raise RuntimeError(f"{sample_id} {spec.target}: OER input FP32/A4 shape mismatch")
    if x_fp32_full.ndim != 2:
        raise RuntimeError(f"{sample_id} {spec.target}: OER input activation must be [T,C]")
    token_count = int(x_fp32_full.shape[0])
    feature_dim = int(x_fp32_full.shape[1])
    if feature_dim != int(oracle_row.feature_dim):
        raise RuntimeError(
            f"{sample_id} {spec.target}: OER feature_dim mismatch: "
            f"oracle={oracle_row.feature_dim}, current={feature_dim}"
        )
    current_valid_count = int(valid_mask_full.sum().item())
    if int(oracle_row.valid_token_count) >= 0 and int(oracle_row.valid_token_count) != current_valid_count:
        raise RuntimeError(
            f"{sample_id} {spec.target}: OER valid_token_count mismatch: "
            f"oracle={oracle_row.valid_token_count}, current={current_valid_count}"
        )
    current_valid_scalar_count = current_valid_count * feature_dim
    if int(oracle_row.valid_scalar_count) >= 0 and int(oracle_row.valid_scalar_count) != current_valid_scalar_count:
        raise RuntimeError(
            f"{sample_id} {spec.target}: OER valid_scalar_count mismatch: "
            f"oracle={oracle_row.valid_scalar_count}, current={current_valid_scalar_count}"
        )

    flat = oracle_row.scalar_indices.detach().cpu().long().reshape(-1)
    if int(flat.numel()) != int(oracle_row.topk_k):
        raise RuntimeError(f"{sample_id} {spec.target}: OER scalar K changed before aggregation")
    if int(torch.unique(flat).numel()) != int(flat.numel()):
        raise RuntimeError(f"{sample_id} {spec.target}: OER scalar oracle contains duplicate indices")
    if bool((flat < 0).any().item()) or bool((flat >= token_count * feature_dim).any().item()):
        raise RuntimeError(f"{sample_id} {spec.target}: OER scalar index out of range")

    scalar_tokens = torch.div(flat, feature_dim, rounding_mode="floor").long()
    scalar_channels = flat.remainder(feature_dim).long()
    if bool((scalar_tokens < 0).any().item()) or bool((scalar_tokens >= token_count).any().item()):
        raise RuntimeError(f"{sample_id} {spec.target}: OER token index out of range")
    if bool((scalar_channels < 0).any().item()) or bool((scalar_channels >= feature_dim).any().item()):
        raise RuntimeError(f"{sample_id} {spec.target}: OER channel index out of range")
    if bool((~valid_mask_full.index_select(0, scalar_tokens)).any().item()):
        raise RuntimeError(f"{sample_id} {spec.target}: OER scalar oracle includes padding token")

    counts_full = torch.zeros(token_count, dtype=torch.double)
    abs_error_full = torch.zeros(token_count, dtype=torch.double)
    delta_x = x_a4_full - x_fp32_full
    abs_selected = torch.abs(delta_x[scalar_tokens, scalar_channels]).double()
    counts_full.scatter_add_(0, scalar_tokens, torch.ones_like(abs_selected, dtype=torch.double))
    abs_error_full.scatter_add_(0, scalar_tokens, abs_selected)
    if int(counts_full.sum().item()) != int(oracle_row.topk_k):
        raise RuntimeError(f"{sample_id} {spec.target}: OER token aggregation did not preserve scalar K")

    expected_tokens = torch.nonzero(valid_mask_full, as_tuple=False).flatten().cpu().long()
    if not torch.equal(expected_tokens, token_indices.cpu().long()):
        raise RuntimeError(f"{sample_id} {spec.target}: OER and harmful/energy/distortion valid token masks differ")
    return counts_full.index_select(0, token_indices).double(), abs_error_full.index_select(0, token_indices).double()


def _token_detail_and_summary(
    *,
    sample_index: int,
    sample_id: str,
    label_id: int,
    bucket: str,
    spec: single_layer.TargetSpec,
    tensors: Mapping[str, Tensor],
    oracle_row: OerOracleRow,
    m_fp32: float,
    m_injected: float,
    actual_margin_drop: float,
    fp32_pred: int,
    injected_pred: int,
    j_star: int,
    classification_flip: int,
    three_factor_reference: ThreeFactorReferenceRow,
    numerical_tolerance: float,
    strict_three_factor_csv_first_order: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token_indices = tensors["token_indices"].long()
    h = tensors["H"].double()
    h_q = tensors["H_q"].double()
    gradient = tensors["G"].double()
    if h.ndim != 2 or h_q.ndim != 2 or gradient.ndim != 2:
        raise RuntimeError(f"{sample_id} {spec.target}: valid output tensors must be [valid_tokens, channels]")
    delta_h = h_q - h
    decision_projection = torch.sum(gradient * delta_h, dim=1).double()
    harmful_score = torch.clamp(-decision_projection, min=0.0)
    energy_score = torch.linalg.vector_norm(h, dim=1)
    distortion_score = torch.linalg.vector_norm(delta_h, dim=1)
    oer_count, oer_abs_error = _aggregate_oer_to_tokens(
        oracle_row=oracle_row,
        x_fp32_full=tensors["X_fp32_full"],
        x_a4_full=tensors["X_a4_full"],
        valid_mask_full=tensors["valid_mask_full"],
        token_indices=token_indices,
        spec=spec,
        sample_id=sample_id,
    )
    valid_token_count = int(token_indices.numel())
    k_eff = _fixed_token_k(valid_token_count)

    ranks_harmful = _ordinal_ranks(harmful_score, token_indices)
    ranks_energy = _ordinal_ranks(energy_score, token_indices)
    ranks_distortion = _ordinal_ranks(distortion_score, token_indices)
    ranks_oer = _ordinal_ranks(oer_count, token_indices)

    top_harmful = _topk_token_set(harmful_score, token_indices, k_eff)
    top_energy = _topk_token_set(energy_score, token_indices, k_eff)
    top_distortion = _topk_token_set(distortion_score, token_indices, k_eff)
    top_oer = _topk_token_set(oer_count, token_indices, k_eff)

    sum_projection = float(decision_projection.sum().item())
    factor_reference = three_factor._factor_metrics(
        h_fp32=tensors["H_batch"],
        h_q=tensors["H_q_batch"],
        gradient=tensors["G_batch"],
        token_padding_mask=tensors["token_padding_mask"],
        spec=spec,
        sample_index=int(tensors["batch_sample_index"].item()),
        numerical_tolerance=float(numerical_tolerance),
    )
    first_order = float(factor_reference["first_order_margin_change"])
    reference_abs_error = abs(sum_projection - first_order)
    if reference_abs_error >= float(numerical_tolerance):
        raise RuntimeError(
            f"{sample_id} {spec.target}: sum_t C_t differs from three-factor factor computation: "
            f"sum={sum_projection:.8g}, three_factor={first_order:.8g}, "
            f"abs_error={reference_abs_error:.8g}"
        )

    csv_abs_error = abs(sum_projection - float(three_factor_reference.first_order_margin_change))
    csv_within_tolerance = csv_abs_error < float(numerical_tolerance)
    if strict_three_factor_csv_first_order and not csv_within_tolerance:
        raise RuntimeError(
            f"{sample_id} {spec.target}: sum_t C_t differs from three-factor CSV: "
            f"sum={sum_projection:.8g}, csv={three_factor_reference.first_order_margin_change:.8g}, "
            f"abs_error={csv_abs_error:.8g}"
        )
    three_factor_actual_error = abs(actual_margin_drop - float(three_factor_reference.actual_margin_drop))
    three_factor_classification_fields_match = (
        int(classification_flip) == int(three_factor_reference.classification_flip)
        and int(j_star) == int(three_factor_reference.j_star)
        and int(fp32_pred) == int(three_factor_reference.fp32_pred)
        and int(injected_pred) == int(three_factor_reference.injected_pred)
    )

    detail_rows: list[dict[str, Any]] = []
    token_list = token_indices.tolist()
    for idx, token_index in enumerate(token_list):
        token_id = int(token_index)
        detail_rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": sample_id,
                "label_id": int(label_id),
                "bucket": bucket,
                "target": spec.target,
                "token_index": token_id,
                "valid_token": True,
                "fp32_energy": float(energy_score[idx].item()),
                "perturbation_norm": float(distortion_score[idx].item()),
                "decision_projection_C": float(decision_projection[idx].item()),
                "harmful_score": float(harmful_score[idx].item()),
                "oer_count": int(oer_count[idx].item()),
                "oer_abs_error": float(oer_abs_error[idx].item()),
                "rank_harmful": int(ranks_harmful[idx].item()),
                "rank_energy": int(ranks_energy[idx].item()),
                "rank_distortion": int(ranks_distortion[idx].item()),
                "rank_oer": int(ranks_oer[idx].item()),
                "is_topk_harmful": int(token_id in top_harmful),
                "is_topk_energy": int(token_id in top_energy),
                "is_topk_distortion": int(token_id in top_distortion),
                "is_topk_oer": int(token_id in top_oer),
            }
        )

    summary_row = {
        "sample_index": int(sample_index),
        "sample_id": sample_id,
        "label_id": int(label_id),
        "bucket": bucket,
        "target": spec.target,
        "fp32_pred": int(fp32_pred),
        "injected_pred": int(injected_pred),
        "j_star": int(j_star),
        "m_fp32": float(m_fp32),
        "m_injected": float(m_injected),
        "classification_flip": int(classification_flip),
        "valid_token_count": valid_token_count,
        "K_eff": int(k_eff),
        "harmful_energy_overlap": _overlap(top_harmful, top_energy, k_eff),
        "harmful_distortion_overlap": _overlap(top_harmful, top_distortion, k_eff),
        "harmful_oer_overlap": _overlap(top_harmful, top_oer, k_eff),
        "harmful_energy_spearman": _spearman_scores(harmful_score, energy_score),
        "harmful_distortion_spearman": _spearman_scores(harmful_score, distortion_score),
        "harmful_oer_spearman": _spearman_scores(harmful_score, oer_count),
        "sum_decision_projection": sum_projection,
        "first_order_margin_change": first_order,
        "predicted_margin_drop": float(-sum_projection),
        "actual_margin_drop": float(actual_margin_drop),
        "three_factor_first_order": float(three_factor_reference.first_order_margin_change),
        "three_factor_actual_margin_drop": float(three_factor_reference.actual_margin_drop),
        "three_factor_reference_abs_error": reference_abs_error,
        "three_factor_csv_abs_error": csv_abs_error,
        "three_factor_csv_within_tolerance": bool(csv_within_tolerance),
        "three_factor_actual_margin_drop_abs_error": three_factor_actual_error,
        "three_factor_actual_margin_drop_within_tolerance": bool(three_factor_actual_error < float(numerical_tolerance)),
        "three_factor_classification_fields_match": bool(three_factor_classification_fields_match),
        "oer_scalar_k": int(oracle_row.topk_k),
        "oer_topk_ratio": float(oracle_row.topk_ratio),
    }
    return detail_rows, summary_row


def _run_target_token_ranking(
    *,
    spec: single_layer.TargetSpec,
    checkpoint_path: Path,
    qparams: Mapping[str, Any],
    loader: Any,
    device: torch.device,
    metadata: Mapping[str, Mapping[str, Any]],
    oer_oracle: OerOracleIndex,
    three_factor_reference: Mapping[tuple[str, str], ThreeFactorReferenceRow],
    numerical_tolerance: float,
    strict_three_factor_csv_first_order: bool = True,
) -> TargetTokenRankingResult:
    fp32_model, _fp32_checkpoint = single_layer._load_clean_fp32_model(
        checkpoint_path,
        device=device,
        context=f"{spec.target}.decision_harmful_fp32_clean_reload",
    )
    reference_fp32_model, _reference_fp32_checkpoint = single_layer._load_clean_fp32_model(
        checkpoint_path,
        device=device,
        context=f"{spec.target}.decision_harmful_single_layer_reference_fp32_clean_reload",
    )
    injected_model, injected_controller, quant_state = single_layer._prepare_injection_model(
        checkpoint_path,
        qparams,
        spec=spec,
        device=device,
    )
    reference_injected_model, reference_injected_controller, _reference_quant_state = single_layer._prepare_injection_model(
        checkpoint_path,
        qparams,
        spec=spec,
        device=device,
    )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sample_order: list[str] = []
    fp32_logits_chunks: list[Tensor] = []
    injected_logits_chunks: list[Tensor] = []
    fp32_pred_chunks: list[Tensor] = []
    injected_pred_chunks: list[Tensor] = []
    sample_offset = 0
    reference_checks = {
        "fp32_logits_identical_to_single_layer_reference": True,
        "injected_logits_identical_to_single_layer_reference": True,
        "fp32_token_padding_mask_identical_to_reference": True,
        "injected_token_padding_mask_identical_to_reference": True,
        "injected_input_fp32_matches_pure_fp32_input": True,
    }

    try:
        for batch in loader:
            _assert_center_crop_protocol(batch)
            sample_ids = single_layer._batch_sample_ids(batch)
            fp32 = _forward_fp32_with_hooks(fp32_model, batch, device=device, spec=spec)
            injected = _forward_injected_with_hooks(injected_model, batch, device=device, spec=spec)
            ref_fp32_logits, ref_labels, ref_fp32_mask = single_layer._forward_logits(reference_fp32_model, batch, device=device)
            ref_injected_logits, ref_injected_labels, ref_injected_mask = single_layer._forward_logits(
                reference_injected_model,
                batch,
                device=device,
            )

            _assert_equal_tensor(f"{spec.target}.fp32_logits_vs_single_layer_reference", fp32["logits"], ref_fp32_logits)
            _assert_equal_tensor(f"{spec.target}.injected_logits_vs_single_layer_reference", injected["logits"], ref_injected_logits)
            _assert_equal_tensor(f"{spec.target}.fp32_labels_vs_reference", fp32["labels"], ref_labels)
            _assert_equal_tensor(f"{spec.target}.injected_labels_vs_reference", injected["labels"], ref_injected_labels)
            _assert_equal_tensor(f"{spec.target}.fp32_token_padding_mask_vs_reference", fp32["token_padding_mask"], ref_fp32_mask)
            _assert_equal_tensor(f"{spec.target}.injected_token_padding_mask_vs_reference", injected["token_padding_mask"], ref_injected_mask)
            _assert_equal_tensor(f"{spec.target}.labels_fp32_vs_injected", fp32["labels"], injected["labels"])
            _assert_equal_tensor(f"{spec.target}.token_padding_mask_fp32_vs_injected", fp32["token_padding_mask"], injected["token_padding_mask"])
            _assert_equal_tensor(f"{spec.target}.target_input_fp32_vs_injected_model_fp32", fp32["X_fp32"], injected["X_fp32_injected"])

            labels: Tensor = fp32["labels"]
            j_star: Tensor = fp32["j_star"]
            m_fp32: Tensor = fp32["m_fp32"]
            m_injected = single_layer._fixed_margin(injected["logits"], labels, j_star)
            fp32_preds: Tensor = fp32["preds"]
            injected_preds: Tensor = injected["preds"]
            fp32_correct: Tensor = fp32["fp32_correct"]

            sample_order.extend(sample_ids)
            fp32_logits_chunks.append(fp32["logits"])
            injected_logits_chunks.append(injected["logits"])
            fp32_pred_chunks.append(fp32_preds)
            injected_pred_chunks.append(injected_preds)

            h_fp32: Tensor = fp32["H"]
            h_q: Tensor = injected["H_q"]
            gradient: Tensor = fp32["G"]
            token_padding_mask: Tensor = fp32["token_padding_mask"]
            x_fp32: Tensor = fp32["X_fp32"]
            x_a4: Tensor = injected["X_a4"]

            for local_index, sample_id in enumerate(sample_ids):
                if not bool(fp32_correct[local_index].item()):
                    continue
                sid = str(sample_id)
                key = (sid, spec.target)
                if key not in oer_oracle.rows_by_key:
                    raise KeyError(f"{sid} {spec.target}: missing fixed OER oracle row")
                if key not in three_factor_reference:
                    raise KeyError(f"{sid} {spec.target}: missing three-factor reference row")

                label_id = int(labels[local_index].item())
                fp32_pred = int(fp32_preds[local_index].item())
                injected_pred = int(injected_preds[local_index].item())
                actual_margin_drop = float((m_fp32[local_index] - m_injected[local_index]).item())
                classification_flip = int(injected_pred != label_id)
                meta = metadata.get(sid, {})
                tensors = _valid_sample_tensors(
                    h_fp32=h_fp32,
                    h_q=h_q,
                    gradient=gradient,
                    x_fp32=x_fp32,
                    x_a4=x_a4,
                    token_padding_mask=token_padding_mask,
                    sample_index=local_index,
                    spec=spec,
                )
                tensors = {
                    **tensors,
                    "H_batch": h_fp32,
                    "H_q_batch": h_q,
                    "G_batch": gradient,
                    "token_padding_mask": token_padding_mask,
                    "batch_sample_index": torch.as_tensor(int(local_index), dtype=torch.long),
                }
                detail, summary = _token_detail_and_summary(
                    sample_index=int(sample_offset + local_index),
                    sample_id=sid,
                    label_id=label_id,
                    bucket=str(meta.get("bucket", "")),
                    spec=spec,
                    tensors=tensors,
                    oracle_row=oer_oracle.rows_by_key[key],
                    m_fp32=float(m_fp32[local_index].item()),
                    m_injected=float(m_injected[local_index].item()),
                    actual_margin_drop=actual_margin_drop,
                    fp32_pred=fp32_pred,
                    injected_pred=injected_pred,
                    j_star=int(j_star[local_index].item()),
                    classification_flip=classification_flip,
                    three_factor_reference=three_factor_reference[key],
                    numerical_tolerance=numerical_tolerance,
                    strict_three_factor_csv_first_order=strict_three_factor_csv_first_order,
                )
                detail_rows.extend(detail)
                summary_rows.append(summary)
            sample_offset += len(sample_ids)

        single_layer._assert_single_target_quant_state(injected_model, spec)
        single_layer._assert_single_target_quant_state(reference_injected_model, spec)
    finally:
        injected_controller.remove_padding_mask_hooks()
        reference_injected_controller.remove_padding_mask_hooks()

    return TargetTokenRankingResult(
        spec=spec,
        sample_order=tuple(sample_order),
        fp32_logits=torch.cat(fp32_logits_chunks, dim=0).contiguous(),
        injected_logits=torch.cat(injected_logits_chunks, dim=0).contiguous(),
        fp32_preds=torch.cat(fp32_pred_chunks, dim=0).contiguous(),
        injected_preds=torch.cat(injected_pred_chunks, dim=0).contiguous(),
        detail_rows=tuple(detail_rows),
        summary_rows=tuple(summary_rows),
        quant_state=tuple(quant_state),
        single_layer_reference_validation=reference_checks,
        checkpoint_clean_reloads=4,
    )


def _assert_shared_fp32_baseline(results: Sequence[TargetTokenRankingResult]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("no target results to validate")
    baseline = results[0]
    for result in results[1:]:
        if result.sample_order != baseline.sample_order:
            raise RuntimeError(f"FP32 sample order differs: {baseline.spec.target} vs {result.spec.target}")
        _assert_equal_tensor(
            f"fp32_logits_shared_baseline.{baseline.spec.target}_vs_{result.spec.target}",
            baseline.fp32_logits,
            result.fp32_logits,
        )
        _assert_equal_tensor(
            f"fp32_predictions_shared_baseline.{baseline.spec.target}_vs_{result.spec.target}",
            baseline.fp32_preds,
            result.fp32_preds,
        )
    return {
        "same_fp32_baseline_across_targets": True,
        "fp32_logits_identical_across_targets": True,
        "fp32_predictions_identical_across_targets": True,
        "sample_order_identical_across_targets": True,
        "sample_order_hash": stable_json_hash(list(baseline.sample_order)),
        "num_eval_samples_seen": int(len(baseline.sample_order)),
    }


def _assert_oer_sample_order(oer_oracle: OerOracleIndex, sample_order: Sequence[str]) -> dict[str, Any]:
    if not oer_oracle.sample_id_order:
        return {"oer_oracle_sample_order_available": False}
    current = tuple(str(item) for item in sample_order)
    if tuple(oer_oracle.sample_id_order) != current:
        raise RuntimeError(
            "OER oracle sample order differs from eval loader order; this script refuses to realign by sorting"
        )
    return {
        "oer_oracle_sample_order_available": True,
        "oer_oracle_sample_order_matches_eval": True,
        "oer_oracle_sample_order_hash": stable_json_hash(list(oer_oracle.sample_id_order)),
    }


def _rows_for_group(rows: Sequence[Mapping[str, Any]], target: str, group: str, flip_status: str) -> list[Mapping[str, Any]]:
    target_rows = [row for row in rows if str(row.get("target")) == str(target)]
    if group == "tail":
        target_rows = [row for row in target_rows if str(row.get("bucket", "")).lower() in {"medium", "few"}]
    elif group != "overall":
        target_rows = [row for row in target_rows if str(row.get("bucket", "")).lower() == group]
    if flip_status == "flip":
        return [row for row in target_rows if int(row.get("classification_flip", 0)) == 1]
    if flip_status == "nonflip":
        return [row for row in target_rows if int(row.get("classification_flip", 0)) == 0]
    return target_rows


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_float(row.get(key, float("nan")))
        if math.isfinite(value):
            values.append(value)
    return values


def _aggregate_metric(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, float, float]:
    values = _finite_values(rows, key)
    if not values:
        return float("nan"), float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.median(array)), float(np.std(array, ddof=0))


def _build_group_summary_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "harmful_energy_overlap",
        "harmful_distortion_overlap",
        "harmful_oer_overlap",
        "harmful_energy_spearman",
        "harmful_distortion_spearman",
        "harmful_oer_spearman",
    )
    result: list[dict[str, Any]] = []
    for spec in single_layer.TARGET_SPECS:
        for group in GROUPS:
            for flip_status in FLIP_STATUSES:
                rows = _rows_for_group(summary_rows, spec.target, group, flip_status)
                payload: dict[str, Any] = {
                    "target": spec.target,
                    "group": group,
                    "flip_status": flip_status,
                    "n": int(len(rows)),
                    "flip_count": int(sum(int(row.get("classification_flip", 0)) for row in rows)),
                    "flip_rate": float(sum(int(row.get("classification_flip", 0)) for row in rows) / len(rows)) if rows else float("nan"),
                }
                for metric in metrics:
                    mean, median, std = _aggregate_metric(rows, metric)
                    payload[f"mean_{metric}"] = mean
                    payload[f"median_{metric}"] = median
                    payload[f"std_{metric}"] = std
                result.append(payload)
    return result


def _build_manifest(
    *,
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    qparams_by_target: Mapping[single_layer.TargetSpec, Mapping[str, Any]],
    oer_oracle: OerOracleIndex,
    results: Sequence[TargetTokenRankingResult],
    shared_fp32_validation: Mapping[str, Any],
    oer_sample_order_validation: Mapping[str, Any],
) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    for result in results:
        spec = result.spec
        qparams = qparams_by_target[spec]
        targets.append(
            {
                "target": spec.target,
                "module_path": spec.module_path,
                "layer_idx": int(spec.layer_idx),
                "module_name": spec.module_name,
                "isolated_w4a4": True,
                "representation_hook": {
                    "position": REPRESENTATION_HOOK_POSITION,
                    "semantic": "target Linear module output",
                },
                "oer_hook": {
                    "position": OER_HOOK_POSITION,
                    "semantic": "target FFN input activation",
                    "oracle_unit": "scalar",
                    "aggregation": "scalar OER indices are aggregated to token-level oer_count before overlap/rank comparison",
                },
                "full_w4a4_forward_used": False,
                "quant_state": list(result.quant_state),
                "qparams_identity": qparams_identity_manifest(qparams, source_hint=spec.target),
                "single_layer_reference_validation": dict(result.single_layer_reference_validation),
                "checkpoint_clean_reloads": int(result.checkpoint_clean_reloads),
            }
        )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_only": True,
        "dataset": single_layer.DEFAULT_DATASET,
        "checkpoint": str(paths["checkpoint"].resolve(strict=False)),
        "checkpoint_name_required": single_layer.DEFAULT_CHECKPOINT_NAME,
        "checkpoint_sha256": single_layer._file_sha256(paths["checkpoint"]),
        "analysis_split": "eval",
        "eval_csv": file_identity(paths["eval_csv"]),
        "test_touched": False,
        "test_csv_not_read": True,
        "train_touched": False,
        "training_performed": False,
        "recalibration_performed": False,
        "full_w4a4_forward_used": False,
        "full_w4a4_intermediate_activation_used_as_injection_input": False,
        "same_waveform": True,
        "same_center_crop": True,
        "center_crop_verified": True,
        "crop_mode": "center",
        "same_padding_mask": True,
        "same_pooling": "mask_aware_mean",
        "same_classifier": True,
        "fixed_competitor": True,
        "competitor_definition": "j_star = argmax_{j != y} z_FP32[j]",
        "main_cohort": "FP32-correct eval samples",
        "groups": list(GROUPS),
        "tail_definition": "tail = medium + few",
        "flip_status_groups": list(FLIP_STATUSES),
        "medium_few_interpretation": "exploratory_only_on_local_data",
        "token_topk": {
            "source": "project_config.TOKEN_TOPK_RATIO",
            "topk_ratio": float(_fixed_token_topk_ratio()),
            "k_policy": "ceil(valid_token_count * TOKEN_TOPK_RATIO), clipped to valid_token_count",
            "k_sweep_performed": False,
            "current_results_used_to_choose_k": False,
        },
        "oer_oracle": {
            "path": str(oer_oracle.path.resolve(strict=False)),
            "manifest_path": None if oer_oracle.manifest_path is None else str(oer_oracle.manifest_path.resolve(strict=False)),
            "oracle_source": "fixed baseline FP32 OER oracle",
            "protocol_version": oer_oracle.protocol_version,
            "source_identity": file_identity(oer_oracle.path),
            "manifest_identity": file_identity(oer_oracle.manifest_path) if oer_oracle.manifest_path is not None else None,
            "hook_position": OER_HOOK_POSITION,
            "activation_position": "input",
            "granularity": "scalar",
            "scalar_index_coordinate_system": "canonical BTC per sample: scalar_index = token_idx * feature_dim + channel_idx",
            "topk_ratios_seen": list(oer_oracle.topk_ratios),
            "shard_count": int(len(oer_oracle.shard_paths)),
            "shard_paths": list(oer_oracle.shard_paths),
            "rebuilt": False,
            "reselected": False,
            "protocol_verified": True,
        },
        "three_factor_reference": {
            "sample_csv": str(paths["three_factor_sample_csv"].resolve(strict=False)),
            "sample_csv_identity": file_identity(paths["three_factor_sample_csv"]),
            "first_order_consistency_checked": True,
            "actual_margin_drop_consistency_checked": False,
            "actual_margin_drop_difference_recorded": True,
            "classification_field_difference_recorded": True,
        },
        "quantization": {
            "scope": "isolated_single_ffn_module_per_target",
            "mode": single_layer.QUANT_MODE.upper(),
            "activation": {
                "bits": 4,
                "granularity": "per_tensor",
                "symmetric": True,
                "qmin": single_layer.EXPECTED_QRANGE[0],
                "qmax": single_layer.EXPECTED_QRANGE[1],
                "zero_point": 0,
            },
            "weight": {
                "bits": 4,
                "granularity": "per_output_channel",
                "symmetric": True,
                "qmin": single_layer.EXPECTED_QRANGE[0],
                "qmax": single_layer.EXPECTED_QRANGE[1],
                "zero_point": 0,
            },
            "non_target_layers_quantized": False,
            "target_predecessor_layers_quantized": False,
        },
        "representation_definition": {
            "H_t": "FP32 target Linear module output",
            "Hq_t": "isolated W4A4 target Linear module output",
            "delta_H_t": "Hq_t - H_t",
            "G_t": "FP32 gradient of fixed margin with respect to H_t",
            "C_t": "sum_channel(G_t * delta_H_t)",
            "harmful_score_t": "max(0, -C_t)",
            "padding_policy": "same valid token mask for harmful, energy, distortion, and OER token aggregation; padding tokens excluded",
        },
        "rankings": {
            "harmful": "harmful_score_t descending; tie-break token_index ascending",
            "energy": "FP32 representation token L2 norm descending",
            "distortion": "output perturbation token L2 norm descending",
            "oer": "token-level oer_count descending after scalar oracle aggregation; tie-break token_index ascending",
            "spearman": "full valid token score Spearman with average tie ranks",
        },
        "targets": targets,
        "validations": {
            **dict(shared_fp32_validation),
            **dict(oer_sample_order_validation),
            "sum_token_decision_projection_matches_three_factor_first_order": True,
            "only_target_module_w4a4_per_experiment": True,
            "same_valid_token_mask_for_harmful_energy_distortion": True,
            "center_crop_verified": True,
            "oer_oracle_protocol_verified": True,
            "oer_scalar_indices_aggregated_to_token_level_before_overlap": True,
            "oer_oracle_reselected": False,
            "token_k_reselected": False,
            "test_touched": False,
        },
        "qparams_root": str(paths["qparams_root"].resolve(strict=False)),
        "calibration_csv": file_identity(paths["calibration_csv"]),
        "calibration_manifest": file_identity(paths["calibration_manifest"]),
        "numerical_tolerance": float(args.numerical_tolerance),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "outputs": {
            "detail_csv": str((paths["output_dir"] / DETAIL_FILENAME).resolve(strict=False)),
            "summary_csv": str((paths["output_dir"] / SUMMARY_FILENAME).resolve(strict=False)),
            "group_summary_csv": str((paths["output_dir"] / GROUP_SUMMARY_FILENAME).resolve(strict=False)),
            "manifest_json": str((paths["output_dir"] / MANIFEST_FILENAME).resolve(strict=False)),
        },
        "disclaimer": (
            "This descriptive ranking does not call high-energy tokens decision-critical, "
            "does not call harmful ranking causal token ranking, and does not propose a training loss from overlap."
        ),
    }


def _print_report(
    *,
    paths: Mapping[str, Path],
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    print("[Decision-Harmful Token Ranking]")
    print(f"checkpoint: {paths['checkpoint']}")
    print(f"oer_oracle: {paths['oer_oracle']}")
    print(f"three_factor_reference: {paths['three_factor_sample_csv']}")
    print(f"output_dir: {paths['output_dir']}")
    print("test_touched: False")
    for spec in single_layer.TARGET_SPECS:
        rows = [row for row in summary_rows if row.get("target") == spec.target]
        flips = int(sum(int(row.get("classification_flip", 0)) for row in rows))
        overlaps = _finite_values(rows, "harmful_oer_overlap")
        mean_oer = float(np.mean(overlaps)) if overlaps else float("nan")
        print(
            f"{spec.target}: n={len(rows)} flips={flips} "
            f"mean_harmful_oer_overlap={_csv_value(mean_oer)}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _assert_center_crop_protocol()
    paths = _resolve_paths(args)
    device = single_layer._device_from_args(args.device)
    metadata = single_layer._load_eval_metadata(paths["eval_csv"])
    qparams_by_target = single_layer._load_qparams_by_target(paths["qparams_root"])
    oer_oracle = _load_oer_oracle_index(paths["oer_oracle"], paths["oer_oracle_manifest"])
    three_factor_reference = _load_three_factor_reference(paths["three_factor_sample_csv"])
    loader = single_layer.build_loader(
        single_layer.dataset_paths(single_layer.DEFAULT_DATASET, None),
        split="eval",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    results: list[TargetTokenRankingResult] = []
    for spec in single_layer.TARGET_SPECS:
        result = _run_target_token_ranking(
            spec=spec,
            checkpoint_path=paths["checkpoint"],
            qparams=qparams_by_target[spec],
            loader=loader,
            device=device,
            metadata=metadata,
            oer_oracle=oer_oracle,
            three_factor_reference=three_factor_reference,
            numerical_tolerance=float(args.numerical_tolerance),
        )
        results.append(result)

    shared_fp32_validation = _assert_shared_fp32_baseline(results)
    oer_sample_order_validation = _assert_oer_sample_order(oer_oracle, results[0].sample_order)
    detail_rows = [row for result in results for row in result.detail_rows]
    summary_rows = [row for result in results for row in result.summary_rows]
    group_summary_rows = _build_group_summary_rows(summary_rows)
    manifest = _build_manifest(
        paths=paths,
        args=args,
        device=device,
        qparams_by_target=qparams_by_target,
        oer_oracle=oer_oracle,
        results=results,
        shared_fp32_validation=shared_fp32_validation,
        oer_sample_order_validation=oer_sample_order_validation,
    )

    output_dir = paths["output_dir"]
    _write_csv(output_dir / DETAIL_FILENAME, detail_rows, DETAIL_COLUMNS)
    _write_csv(output_dir / SUMMARY_FILENAME, summary_rows, SUMMARY_COLUMNS)
    _write_csv(output_dir / GROUP_SUMMARY_FILENAME, group_summary_rows, GROUP_SUMMARY_COLUMNS)
    _write_json(output_dir / MANIFEST_FILENAME, manifest)
    _print_report(paths=paths, summary_rows=summary_rows)
    return {
        "paths": paths,
        "detail_rows": detail_rows,
        "summary_rows": summary_rows,
        "group_summary_rows": group_summary_rows,
        "manifest": manifest,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=single_layer.DEFAULT_DATASET)
    parser.add_argument("--checkpoint", default=None, help="Must resolve to local_data/BEATs_local_data.pt.")
    parser.add_argument("--qparams-root", default=None, help="Directory containing frozen W4A4 qparams.")
    parser.add_argument("--oer-oracle", default=None, help="Fixed baseline FP32 OER oracle .pt. The script only reads it.")
    parser.add_argument("--three-factor-sample-csv", default=None, help="three_factor_sample.csv from src.export.three_factor_decomposition.")
    parser.add_argument("--output-dir", default=None, help="Directory for decision-harmful token ranking CSVs.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=single_layer._default_batch_size())
    parser.add_argument("--num-workers", type=int, default=single_layer._default_num_workers())
    parser.add_argument("--seed", type=int, default=int(single_layer._cfg_get(project_config, "SPLIT_SEED", 42)))
    parser.add_argument("--numerical-tolerance", type=float, default=DEFAULT_NUMERICAL_TOLERANCE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
