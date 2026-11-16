from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

import config as project_config
from src.baseline.export_eval import (
    _build_finetuned_beats,
    _normalise_checkpoint_label_mapping,
)
from src.baseline.quant_checkpoint import qparams_hash
from src.innovation.export_QVF import (
    MECHANISM_EXPORT_LAYERS,
    MECHANISM_QUANT_MODES,
    _collect_mechanism_rows,
    _parse_expected_layer_weights,
    _qvf_weighted_variant,
)
from src.quant.quant_linear import QuantLinear
from src.verify.ffn_token_survival_probe import (
    CSV_FILENAME,
    MANIFEST_FILENAME,
    SHARDS_DIRNAME,
    _load_qparams_if_valid,
    _save_qparams,
    run_token_survival_probe,
)


DEFAULT_BETAS = (0.01, 0.05, 0.10)
TARGET_MODULES = tuple((int(layer_idx), "fc1") for layer_idx in MECHANISM_EXPORT_LAYERS)
CHECKPOINT_NAME = "qvf_stage1_checkpoint.pt"
DRIVER_MANIFEST_NAME = "qvf_token_survival_driver_manifest.json"


def _portable_path_text(value: Path | str) -> str:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _portable_path(value: Path | str) -> Path:
    return Path(_portable_path_text(value))


def _looks_like_windows_absolute(text: str) -> bool:
    return len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] == "/"


def _path_under_dataset_root(value: Path | str, dataset_root: Path) -> Path:
    text = _portable_path_text(value)
    path = Path(text)
    if path.is_absolute():
        return path
    marker = "local_data/"
    if (text.startswith(marker) or _looks_like_windows_absolute(text)) and marker in text:
        return dataset_root / text.split(marker, 1)[1]
    return path


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    path = _portable_path(path)
    if not path.exists():
        raise FileNotFoundError(f"QVF checkpoint missing: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        raise RuntimeError(
            "failed to load QVF checkpoint; the file exists but is not a readable "
            f"torch checkpoint, which usually means an incomplete or corrupted artifact: {path}"
        ) from None
    if not isinstance(payload, Mapping):
        raise TypeError(f"QVF checkpoint must be a mapping, got {type(payload)!r}: {path}")
    return payload


def _file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    resolved = _portable_path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    hasher = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def _path_str(path: Path | str | None) -> str | None:
    return None if path is None else str(_portable_path(path))


def _dataset_name(default: str | None = None) -> str:
    if default:
        return str(default)
    active = getattr(project_config, "ACTIVE_DATASET", None)
    if active:
        return str(active)
    current = getattr(project_config, "CURRENT_DATASET", {})
    if isinstance(current, Mapping):
        for key in ("NAME", "name", "DATASET_NAME", "dataset_name"):
            if current.get(key):
                return str(current[key])
        root = current.get("ROOT")
        if root:
            return _portable_path(root).name
    raise AttributeError("cannot infer dataset name; pass --dataset")


def _dataset_cfg(dataset_name: str) -> dict[str, Any]:
    datasets = getattr(project_config, "DATASETS", {})
    if isinstance(datasets, Mapping) and dataset_name in datasets:
        cfg = copy.deepcopy(dict(datasets[dataset_name]))
    else:
        current = getattr(project_config, "CURRENT_DATASET", {})
        if not isinstance(current, Mapping) or not current:
            raise KeyError(f"dataset {dataset_name!r} not found in config.DATASETS")
        cfg = copy.deepcopy(dict(current))
    cfg.setdefault("NAME", dataset_name)
    return cfg


def _dataset_root(dataset_cfg: Mapping[str, Any]) -> Path:
    root = dataset_cfg.get("ROOT")
    if root is None:
        raise KeyError("dataset config must contain ROOT")
    return _portable_path(root)


def _parse_betas(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        if not parts:
            raise ValueError("--betas cannot be empty")
        betas = tuple(float(item) for item in parts)
    else:
        betas = tuple(float(item) for item in raw)
    for beta in betas:
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be finite and non-negative, got {beta!r}")
    if len(set(betas)) != len(betas):
        raise ValueError(f"duplicate beta values are not allowed: {betas}")
    return betas


def _beta_stem(beta: float) -> str:
    value = float(beta)
    if math.isclose(value, 0.01, rel_tol=0.0, abs_tol=1e-12):
        return "0p01"
    if math.isclose(value, 0.05, rel_tol=0.0, abs_tol=1e-12):
        return "0p05"
    if math.isclose(value, 0.10, rel_tol=0.0, abs_tol=1e-12):
        return "0p10"
    text = f"{value:.10g}"
    return text.replace("-", "m").replace(".", "p")


def _candidate_dir(qvf_root: Path, beta: float) -> Path:
    return qvf_root / f"beta_{_beta_stem(beta)}"


def _str_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _require_close(name: str, actual: Any, expected: float, *, atol: float = 1e-12) -> None:
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=float(atol)):
        raise RuntimeError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if str(actual) != str(expected):
        raise RuntimeError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def _require_weight_payload(name: str, actual: Any, expected: Mapping[str, float]) -> None:
    if not isinstance(actual, Mapping):
        raise RuntimeError(f"{name} must be a mapping, got {type(actual)!r}")
    actual_keys = set(str(key) for key in actual.keys())
    expected_keys = set(str(key) for key in expected.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(f"{name} keys mismatch: actual={sorted(actual_keys)!r}, expected={sorted(expected_keys)!r}")
    for key, expected_value in expected.items():
        _require_close(f"{name}.{key}", actual.get(key), float(expected_value))


def _validate_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    beta: float,
    expected_method_variant: str | None = None,
    expected_protocol_version: str | None = None,
    expected_layer_weights: Mapping[str, float] | None = None,
    expected_weighted_variant: str | None = None,
) -> None:
    required = ("cfg", "model_state_dict", "label_mapping", "qparams", "qparams_hash")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"QVF checkpoint missing required key(s) {missing}: {checkpoint_path}")
    if not _str_bool(checkpoint.get("use_qvf_token_fidelity", False)):
        raise RuntimeError(f"checkpoint is not marked use_qvf_token_fidelity=True: {checkpoint_path}")
    ck_beta = float(checkpoint.get("qvf_beta", beta))
    if abs(ck_beta - float(beta)) > 1e-12:
        raise RuntimeError(f"checkpoint qvf_beta mismatch for {checkpoint_path}: expected {beta}, got {ck_beta}")
    computed = qparams_hash(checkpoint["qparams"])
    stored = str(checkpoint.get("qparams_hash"))
    if computed != stored:
        raise RuntimeError(f"checkpoint qparams_hash mismatch: stored={stored}, computed={computed}")
    if expected_method_variant:
        _require_equal("checkpoint.method_variant", checkpoint.get("method_variant"), expected_method_variant)
    protocol = checkpoint.get("qvf_protocol")
    if expected_method_variant or expected_protocol_version or expected_layer_weights:
        if not isinstance(protocol, Mapping):
            raise RuntimeError(f"checkpoint qvf_protocol must be a mapping: {checkpoint_path}")
    if isinstance(protocol, Mapping):
        if expected_method_variant:
            _require_equal("checkpoint.qvf_protocol.method_variant", protocol.get("method_variant"), expected_method_variant)
        if expected_protocol_version:
            _require_equal("checkpoint.qvf_protocol.protocol_version", protocol.get("protocol_version"), expected_protocol_version)
        _require_close("checkpoint.qvf_protocol.qvf_beta", protocol.get("qvf_beta", beta), float(beta))
    if expected_layer_weights:
        _require_equal("checkpoint.qvf_layer_weight_policy", checkpoint.get("qvf_layer_weight_policy"), "weighted_sensitive_layer_mean")
        if not isinstance(protocol, Mapping):
            raise RuntimeError(f"checkpoint qvf_protocol must be a mapping: {checkpoint_path}")
        _require_equal("checkpoint.qvf_protocol.layer_weight_policy", protocol.get("layer_weight_policy"), "weighted_sensitive_layer_mean")
        if expected_weighted_variant:
            _require_equal("checkpoint.qvf_weighted_variant", checkpoint.get("qvf_weighted_variant"), expected_weighted_variant)
            _require_equal("checkpoint.qvf_protocol.qvf_weighted_variant", protocol.get("qvf_weighted_variant"), expected_weighted_variant)
        _require_weight_payload("checkpoint.sensitive_layer_weights", checkpoint.get("sensitive_layer_weights"), expected_layer_weights)
        _require_weight_payload("checkpoint.qvf_layer_weights", checkpoint.get("qvf_layer_weights"), expected_layer_weights)
        _require_weight_payload("checkpoint.qvf_protocol.sensitive_layer_weights", protocol.get("sensitive_layer_weights"), expected_layer_weights)
        _require_weight_payload("checkpoint.qvf_protocol.qvf_layer_weights", protocol.get("qvf_layer_weights"), expected_layer_weights)


def _clean_state_dict_from_quantized(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    quant_buffer_suffixes = (
        ".activation_scale",
        ".activation_zero_point",
        ".weight_scale",
        ".weight_zero_point",
    )
    for key, value in state_dict.items():
        key_text = str(key)
        if key_text.endswith(quant_buffer_suffixes):
            continue
        new_key = key_text
        for module_name in ("fc1", "fc2"):
            needle = f".{module_name}.linear."
            replacement = f".{module_name}."
            if needle in new_key:
                new_key = new_key.replace(needle, replacement)
        if new_key in clean:
            raise RuntimeError(f"state_dict key collision while dequantizing QVF checkpoint: {new_key}")
        clean[new_key] = value
    return clean


def _build_clean_qvf_model(checkpoint: Mapping[str, Any]) -> nn.Module:
    label_mapping = _normalise_checkpoint_label_mapping(checkpoint.get("label_mapping"))
    model = _build_finetuned_beats(checkpoint["cfg"], label_mapping)
    clean_state = _clean_state_dict_from_quantized(checkpoint["model_state_dict"])
    model.load_state_dict(clean_state, strict=True)
    quant_modules = [name for name, module in model.named_modules() if isinstance(module, QuantLinear)]
    if quant_modules:
        raise RuntimeError(f"clean QVF restore unexpectedly contains QuantLinear modules: {quant_modules[:5]}")
    model.eval()
    return model


@dataclass
class QVFProbeConfig:
    base: Any
    dataset_name: str
    current_dataset: Mapping[str, Any]
    qvf_checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    output_dir: Path
    device: str
    seed: int
    eval_batch_size: int | None = None
    calibration_batch_size: int | None = None
    num_workers: int | None = None
    write_combined_sample_csv: bool = False
    token_topk_ratio: float | None = None
    _probe_cfg: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        inherited = getattr(self.base, "FFN_TOKEN_SURVIVAL_PROBE", {})
        if not isinstance(inherited, Mapping):
            inherited = {}
        probe = copy.deepcopy(dict(inherited))
        probe.update(
            {
                "OUTPUT_DIR": str(self.output_dir),
                "TARGET_MODULES": list(TARGET_MODULES),
                "INPUT_ACTIVATION_TARGET_MODULES": list(TARGET_MODULES),
                "SENSITIVE_TARGET_MODULES": list(TARGET_MODULES),
                "QUANT_MODES": list(MECHANISM_QUANT_MODES),
                "OPTIONAL_CONTROL_QUANT_MODES": [],
                "INCLUDE_W4_ONLY_CONTROL": False,
                "RUN_BLOCK_OUTPUT_TRACE": False,
                "RUN_INPUT_ACTIVATION_TRACE": True,
                "RUN_OUTPUT_PROJECTION_TRACE": False,
                "STREAMING_SUMMARY": True,
                "WRITE_COMBINED_SAMPLE_CSV": bool(self.write_combined_sample_csv),
                "HASH_ARTIFACT_FILES": True,
                "DEVICE": str(self.device),
                "SEED": int(self.seed),
            }
        )
        if self.eval_batch_size is not None:
            probe["EVAL_BATCH_SIZE"] = int(self.eval_batch_size)
        if self.calibration_batch_size is not None:
            probe["CALIBRATION_BATCH_SIZE"] = int(self.calibration_batch_size)
        if self.num_workers is not None:
            probe["NUM_WORKERS"] = int(self.num_workers)
        if self.token_topk_ratio is not None:
            probe["TOKEN_TOPK_RATIO"] = float(self.token_topk_ratio)
        self._probe_cfg = probe

    @property
    def FFN_TOKEN_SURVIVAL_PROBE(self) -> Mapping[str, Any]:
        return self._probe_cfg

    @property
    def CURRENT_DATASET(self) -> Mapping[str, Any]:
        return self.current_dataset

    @property
    def ACTIVE_DATASET(self) -> str:
        return self.dataset_name

    @property
    def CHECKPOINT_PATH(self) -> str:
        return str(self.qvf_checkpoint_path)

    @property
    def MODEL_CHECKPOINT(self) -> str:
        return str(self.qvf_checkpoint_path)

    @property
    def DEVICE(self) -> str:
        return str(self.device)

    @property
    def SEED(self) -> int:
        return int(self.seed)

    @property
    def BEATS(self) -> Mapping[str, Any]:
        beats = copy.deepcopy(dict(getattr(self.base, "BEATS", {})))
        beats["CHECKPOINT"] = str(self.qvf_checkpoint_path)
        return beats

    def restore_clean_model(self, _cfg: Any = None) -> nn.Module:
        return _build_clean_qvf_model(self.checkpoint)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.base, key)


def _infer_token_topk_ratio(checkpoint: Mapping[str, Any]) -> float | None:
    qvf_protocol = checkpoint.get("qvf_protocol")
    if isinstance(qvf_protocol, Mapping):
        for key in ("topk_ratio", "token_topk_ratio"):
            value = qvf_protocol.get(key)
            if value not in (None, ""):
                return float(value)
    training_config = checkpoint.get("kd_config")
    if isinstance(training_config, Mapping):
        value = training_config.get("qvf_topk_ratio") or training_config.get("topk_ratio")
        if value not in (None, ""):
            return float(value)
    return None


def _seed_checkpoint_qparams(cfg: QVFProbeConfig, *, result_dir: Path) -> list[dict[str, Any]]:
    qparams = cfg.checkpoint.get("qparams")
    if not isinstance(qparams, Mapping):
        raise TypeError("QVF checkpoint qparams must be a mapping")
    records: list[dict[str, Any]] = []
    for layer_idx, module_name in TARGET_MODULES:
        module_path = f"encoder.layers.{int(layer_idx)}.{module_name}"
        target_qparams = qparams.get(module_path)
        if not isinstance(target_qparams, Mapping):
            raise KeyError(f"QVF checkpoint qparams missing {module_path}")
        if "activation" not in target_qparams or "weight" not in target_qparams:
            raise KeyError(f"QVF checkpoint qparams[{module_path}] must contain activation and weight")
        action = "written_from_qvf_checkpoint"
        cached = _load_qparams_if_valid(result_dir, int(layer_idx), module_name, cfg=cfg)
        if cached is not None:
            cached_qparams, cached_path = cached
            if qparams_hash(cached_qparams) == qparams_hash(target_qparams):
                qparams_path = cached_path
                action = "reused_valid_qvf_checkpoint_qparams_cache"
            else:
                qparams_path = _save_qparams(result_dir, int(layer_idx), module_name, target_qparams, cfg=cfg)
                action = "replaced_mismatched_qparams_cache"
        else:
            qparams_path = _save_qparams(result_dir, int(layer_idx), module_name, target_qparams, cfg=cfg)
        records.append(
            {
                "layer_idx": int(layer_idx),
                "module_name": module_name,
                "module_path": module_path,
                "qparams_cache": str(qparams_path),
                "qparams_cache_sha256": _file_sha256(qparams_path),
                "source": "qvf_checkpoint_frozen_qparams",
                "action": action,
            }
        )
    return records


def _expected_shards(result_dir: Path) -> dict[str, list[str]]:
    shard_dir = result_dir / SHARDS_DIRNAME
    status: dict[str, list[str]] = {}
    for quant_mode in MECHANISM_QUANT_MODES:
        for layer_idx, module_name in TARGET_MODULES:
            pattern = (
                f"eval__{quant_mode}__intervene_L{int(layer_idx):02d}_{module_name}"
                f"__trace_L{int(layer_idx):02d}_{module_name}_input_activation__shard*.csv"
            )
            matches = sorted(shard_dir.glob(pattern))
            status[pattern] = [str(path) for path in matches]
    return status


def _validate_generated_mechanism(
    *,
    result_dir: Path,
    dataset_name: str,
    seed: int,
    beta: float,
) -> dict[str, Any]:
    expected = _expected_shards(result_dir)
    missing = [pattern for pattern, matches in expected.items() if not matches]
    if missing:
        raise FileNotFoundError(f"missing QVF token_survival mechanism shard(s): {missing}")
    rows, manifest = _collect_mechanism_rows(
        dataset_name=dataset_name,
        seed=int(seed),
        model_role="qvf_candidate",
        beta=float(beta),
        token_survival_dir=result_dir,
        layers=MECHANISM_EXPORT_LAYERS,
        quant_modes=MECHANISM_QUANT_MODES,
    )
    return {
        "status": "validated_for_export_QVF",
        "mechanism_rows": len(rows),
        "source_kind": manifest.get("source_kind"),
        "source_files": manifest.get("source_files", []),
        "expected_shards": expected,
    }


def _build_probe_config(
    *,
    dataset_name: str,
    dataset_cfg: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> QVFProbeConfig:
    return QVFProbeConfig(
        base=project_config,
        dataset_name=dataset_name,
        current_dataset=dataset_cfg,
        qvf_checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=str(args.device),
        seed=int(args.seed),
        eval_batch_size=args.eval_batch_size,
        calibration_batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
        write_combined_sample_csv=bool(args.write_combined_sample_csv),
        token_topk_ratio=_infer_token_topk_ratio(checkpoint),
    )


def run_qvf_token_survival(args: argparse.Namespace) -> dict[str, Any]:
    dataset_name = _dataset_name(args.dataset)
    dataset_cfg = _dataset_cfg(dataset_name)
    dataset_root = _dataset_root(dataset_cfg)
    qvf_root = _path_under_dataset_root(args.qvf_root, dataset_root) if args.qvf_root else dataset_root / "innovation" / "QVF" / f"seed_{int(args.seed)}"
    betas = _parse_betas(args.betas)
    expected_layer_weights = _parse_expected_layer_weights(args.expected_layer_weights)
    expected_weighted_variant = str(args.expected_qvf_weighted_variant or "").strip()
    if expected_layer_weights and not expected_weighted_variant:
        expected_weighted_variant = _qvf_weighted_variant(expected_layer_weights)
    expected_method_variant = str(args.expected_method_variant or "").strip() or None
    expected_protocol_version = str(args.expected_protocol_version or "").strip() or None
    root_summary: dict[str, Any] = {
        "status": "completed",
        "script": "src.innovation.export_qvf_token_survival",
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "seed": int(args.seed),
        "qvf_root": str(qvf_root),
        "betas": [float(beta) for beta in betas],
        "expected_method_variant": expected_method_variant,
        "expected_protocol_version": expected_protocol_version,
        "expected_qvf_weighted_variant": expected_weighted_variant,
        "expected_layer_weights": dict(expected_layer_weights or {}),
        "target_modules": [f"L{layer_idx}.{module_name}" for layer_idx, module_name in TARGET_MODULES],
        "quant_modes": list(MECHANISM_QUANT_MODES),
        "split_policy": {"analysis_split": "eval", "test_touched": False},
        "candidates": {},
    }

    for beta in betas:
        candidate_dir = _candidate_dir(qvf_root, float(beta))
        checkpoint_path = candidate_dir / str(args.checkpoint_name)
        result_dir = candidate_dir / "token_survival"
        candidate_manifest: dict[str, Any] = {
            "status": "planned" if args.dry_run else "running",
            "dataset_name": dataset_name,
            "seed": int(args.seed),
            "qvf_beta": float(beta),
            "candidate_dir": str(candidate_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_exists": checkpoint_path.exists(),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "expected_method_variant": expected_method_variant,
            "expected_protocol_version": expected_protocol_version,
            "expected_qvf_weighted_variant": expected_weighted_variant,
            "expected_layer_weights": dict(expected_layer_weights or {}),
            "token_survival_dir": str(result_dir),
            "target_modules": [f"L{layer_idx}.{module_name}" for layer_idx, module_name in TARGET_MODULES],
            "quant_modes": list(MECHANISM_QUANT_MODES),
            "analysis_split": "eval",
            "calibration_split": "calibration",
            "test_touched": False,
            "write_combined_sample_csv": bool(args.write_combined_sample_csv),
            "token_topk_ratio": None,
        }
        if args.dry_run:
            print(f"[qvf-token-survival] dry-run beta={beta}: {result_dir}")
            root_summary["candidates"][str(beta)] = candidate_manifest
            continue

        checkpoint = _load_checkpoint(checkpoint_path)
        _validate_checkpoint_payload(
            checkpoint,
            checkpoint_path=checkpoint_path,
            beta=float(beta),
            expected_method_variant=expected_method_variant,
            expected_protocol_version=expected_protocol_version,
            expected_layer_weights=expected_layer_weights,
            expected_weighted_variant=expected_weighted_variant,
        )
        probe_cfg = _build_probe_config(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            output_dir=result_dir,
            args=args,
        )
        candidate_manifest.update(
            {
                "checkpoint_qvf_beta": float(checkpoint.get("qvf_beta", beta)),
                "checkpoint_qparams_hash": str(checkpoint.get("qparams_hash")),
                "checkpoint_method_variant": str(checkpoint.get("method_variant", "")),
                "token_topk_ratio": _infer_token_topk_ratio(checkpoint),
            }
        )

        print(f"[qvf-token-survival] beta={beta} output={result_dir}")
        qparams_records = _seed_checkpoint_qparams(probe_cfg, result_dir=result_dir)
        probe_result = run_token_survival_probe(probe_cfg)
        validation = _validate_generated_mechanism(
            result_dir=result_dir,
            dataset_name=dataset_name,
            seed=int(args.seed),
            beta=float(beta),
        )
        candidate_manifest.update(
            {
                "status": "completed",
                "qparams_cache": qparams_records,
                "probe_result": {key: str(value) for key, value in probe_result.items()},
                "validation": validation,
                "token_survival_manifest": str(result_dir / MANIFEST_FILENAME),
                "combined_metrics_csv": str(result_dir / CSV_FILENAME),
            }
        )
        _write_json_atomic(result_dir / DRIVER_MANIFEST_NAME, candidate_manifest)
        root_summary["candidates"][str(beta)] = candidate_manifest
        print(
            "[qvf-token-survival] beta="
            f"{beta} validated rows={validation['mechanism_rows']} source={validation['source_kind']}"
        )

    if not args.dry_run:
        _write_json_atomic(qvf_root / "qvf_token_survival_runs_manifest.json", root_summary)
    return root_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-beta QVF token_survival mechanism tables for export_QVF.py."
    )
    parser.add_argument("--dataset", default=None, help="Dataset name from config.DATASETS; defaults to ACTIVE_DATASET.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--betas", default=",".join(f"{beta:.2f}" for beta in DEFAULT_BETAS))
    parser.add_argument("--qvf-root", default=None, help="Defaults to <dataset_root>/innovation/QVF/seed_<seed>.")
    parser.add_argument("--checkpoint-name", default=CHECKPOINT_NAME)
    parser.add_argument("--expected-method-variant", default=None)
    parser.add_argument("--expected-protocol-version", default=None)
    parser.add_argument("--expected-layer-weights", default=None)
    parser.add_argument("--expected-qvf-weighted-variant", default=None)
    parser.add_argument("--device", default=str(getattr(project_config, "DEVICE", "cuda")))
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--calibration-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--write-combined-sample-csv",
        action="store_true",
        help="Also write token_survival_metrics.csv. By default export_QVF reads target shards directly.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_qvf_token_survival(args)
    print(json.dumps({"status": result["status"], "qvf_root": result["qvf_root"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
