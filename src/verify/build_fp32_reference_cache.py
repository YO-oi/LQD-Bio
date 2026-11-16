from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.analysis.build_fp32_oer_oracle import export_fp32_oer_oracle
    from src.analysis.build_fp32_token_trace_oracle import ORACLE_PROTOCOL_VERSION, ORACLE_STORAGE_FORMAT, export_fp32_token_trace_oracle
    from src.analysis.fp32_geometry_sanity_check import FP32GeometrySanityError, run_fp32_geometry_sanity_check
    from src.export.export_fp32_core_cache import (
        _build_eval_loader,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _file_sha256,
        _make_audio_config,
        _dataset_cfg,
        _default_cfg,
        _nested_get,
        _restore_fp32_teacher,
        export_fp32_core_cache,
    )
    from src.export.export_fp32_token_trace_cache import export_fp32_token_trace_cache
except ImportError:  # pragma: no cover - support package-relative debugging.
    from ..analysis.build_fp32_oer_oracle import export_fp32_oer_oracle
    from ..analysis.build_fp32_token_trace_oracle import ORACLE_PROTOCOL_VERSION, ORACLE_STORAGE_FORMAT, export_fp32_token_trace_oracle
    from ..analysis.fp32_geometry_sanity_check import FP32GeometrySanityError, run_fp32_geometry_sanity_check
    from ..export.export_fp32_core_cache import (
        _build_eval_loader,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _file_sha256,
        _make_audio_config,
        _dataset_cfg,
        _default_cfg,
        _nested_get,
        _restore_fp32_teacher,
        export_fp32_core_cache,
    )
    from ..export.export_fp32_token_trace_cache import export_fp32_token_trace_cache

CORE_CACHE_FILENAME = "core_eval.pt"
OER_ORACLE_FILENAME = "oer_oracle_eval.pt"
TOKEN_TRACE_ORACLE_FILENAME = "token_trace_oracle_eval.pt"
TOKEN_TRACE_DIRNAME = "token_trace_eval"
BUNDLE_FILENAME = "fp32_reference_bundle.json"
SANITY_CSV_FILENAME = "fp32_reference_sanity_eval.csv"
SANITY_RAW_REPORT_FILENAME = "fp32_reference_sanity_raw.json"
SANITY_SUMMARY_FILENAME = "fp32_reference_sanity_summary.json"
GEOMETRY_JSON_FILENAME = "fp32_reference_geometry_eval.json"
GEOMETRY_CSV_FILENAME = "fp32_reference_geometry_eval.csv"


def _reference_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_REFERENCE_CACHE", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    reference_cfg = _reference_cfg(cfg)
    if key in reference_cfg:
        return reference_cfg[key]
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
    raise AttributeError("????????????????????ACTIVE_DATASET ??CURRENT_DATASET['ROOT']")


def _eval_csv(cfg: Any) -> Path:
    configured = _setting_get(cfg, "EVAL_CSV", None)
    if configured is not None:
        return Path(configured)
    dataset_value = _nested_get(_dataset_cfg(cfg), "EVAL_CSV", None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError("missing eval csv; configure CURRENT_DATASET['EVAL_CSV'] or FP32_REFERENCE_CACHE['EVAL_CSV']")


def _label_mapping_path(cfg: Any) -> Path | None:
    configured = _setting_get(cfg, "LABEL_MAPPING", None)
    if configured is not None:
        return Path(configured)
    dataset_value = _nested_get(_dataset_cfg(cfg), "LABEL_MAPPING", None)
    if dataset_value is not None:
        return Path(dataset_value)
    return None


def _require_existing_file(path: Path, *, name: str) -> Path:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{name} ????? {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} ??????: {path}")
    return path.resolve()


def _preflight_inputs(cfg: Any) -> tuple[Path, Path, str]:
    """Validate eval csv, label mapping, and FP32 checkpoint inputs."""

    eval_csv = _require_existing_file(_eval_csv(cfg), name="eval.csv")

    label_mapping = _label_mapping_path(cfg)
    if label_mapping is None:
        raise AttributeError("missing label_mapping.json; configure CURRENT_DATASET['LABEL_MAPPING'] or FP32_REFERENCE_CACHE['LABEL_MAPPING']")
    label_mapping = _require_existing_file(label_mapping, name="label_mapping.json")

    checkpoint = _checkpoint_path_for_manifest(cfg)
    if checkpoint is None:
        raise AttributeError("missing FP32 teacher checkpoint; configure BEATS['CHECKPOINT'] or CHECKPOINT_PATH")
    checkpoint_path = _require_existing_file(Path(checkpoint), name="FP32 teacher checkpoint")
    return eval_csv, label_mapping, str(checkpoint_path)


def _cache_root(cfg: Any) -> Path:
    configured = _setting_get(cfg, "CACHE_ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference"


def _reports_root(cfg: Any) -> Path:
    configured = _setting_get(cfg, "REPORTS_ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "reports"


def _force(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "FORCE", False))


def _include_token_trace(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "INCLUDE_TOKEN_TRACE_CACHE", False))


def _strict_sanity(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "STRICT_SANITY", _setting_get(cfg, "STRICT", False)))


def _include_token_trace_oracle(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "INCLUDE_TOKEN_TRACE_ORACLE_CACHE", True))


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audio_config_hash(cfg: Any) -> str:
    audio_config = _make_audio_config(cfg)
    payload = getattr(audio_config, "__dict__", audio_config)
    return _stable_json_hash(payload)


def _relative_artifact_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _manifest_from_result(result: Mapping[str, Any] | None, payload_key: str) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    payload = result.get(payload_key)
    if isinstance(payload, Mapping):
        manifest = payload.get("manifest", {})
        return manifest if isinstance(manifest, Mapping) else {}
    return {}


def _file_identity(path: Path | None) -> str | None:
    return None if path is None else _file_sha256(path)


def _sidecar_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"existing artifact must be a mapping: {path}, got {type(payload)!r}")
    return dict(payload)


def _manifest_shard_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = manifest.get("shards", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _manifest_shard_paths(manifest: Mapping[str, Any]) -> list[str]:
    raw_paths = manifest.get("shard_paths")
    if isinstance(raw_paths, list):
        return [str(item) for item in raw_paths]
    return [str(item.get("path")) for item in _manifest_shard_entries(manifest) if item.get("path") is not None]


def _manifest_shard_sha256(manifest: Mapping[str, Any]) -> dict[str, str | None]:
    raw = manifest.get("shard_sha256")
    if isinstance(raw, Mapping):
        return {str(key): (None if value is None else str(value)) for key, value in raw.items()}
    result: dict[str, str | None] = {}
    for entry in _manifest_shard_entries(manifest):
        path = entry.get("path")
        if path is not None and "sha256" in entry:
            result[str(path)] = None if entry.get("sha256") is None else str(entry.get("sha256"))
    return result


def _resolve_artifact_relative_path(artifact_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return artifact_path.parent / path


def _verify_manifest_shards(artifact_path: Path, manifest: Mapping[str, Any], *, verify_sha256: bool = True) -> list[str]:
    errors: list[str] = []
    shard_paths = _manifest_shard_paths(manifest)
    declared_num_shards = manifest.get("num_shards")
    if declared_num_shards is not None and int(declared_num_shards) != len(shard_paths):
        errors.append(f"num_shards={declared_num_shards} but shard_paths has {len(shard_paths)} entries")
    if bool(manifest.get("sharded", False)) and not shard_paths:
        errors.append("sharded manifest has no shard paths")
    shard_hashes = _manifest_shard_sha256(manifest)
    for shard_rel in shard_paths:
        shard_path = _resolve_artifact_relative_path(artifact_path, shard_rel)
        if not shard_path.exists() or not shard_path.is_file():
            errors.append(f"missing shard: {shard_rel}")
            continue
        expected_hash = shard_hashes.get(str(shard_rel))
        if verify_sha256 and expected_hash:
            actual_hash = _file_sha256(shard_path)
            if actual_hash != expected_hash:
                errors.append(f"shard sha256 mismatch: {shard_rel}")
    return errors


def _ensure_existing_artifact_valid(
    *,
    artifact_path: Path,
    payload_key: str,
    path_key: str,
    expected_manifest: Mapping[str, Any],
    eval_csv: Path,
    checkpoint_path: str | Path | None,
    label_mapping: Path | None = None,
    expected_sample_order_hash: str | None = None,
    verify_shards: bool = True,
) -> Mapping[str, Any] | None:
    if not artifact_path.exists():
        return None
    payload = _load_torch_mapping(artifact_path)
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"existing artifact lacks manifest: {artifact_path}; pass --force to rebuild")
    mismatches: list[str] = []
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"manifest[{key}] expected {expected!r}, got {actual!r}")
    eval_hash = _file_sha256(eval_csv)
    if manifest.get("eval_csv_sha256") != eval_hash:
        mismatches.append("eval_csv_sha256 mismatch")
    checkpoint_hash = _file_sha256(checkpoint_path)
    if checkpoint_hash is not None:
        if manifest.get("checkpoint_sha256") != checkpoint_hash:
            mismatches.append("checkpoint_sha256 mismatch")
        if manifest.get("checkpoint_hash") != checkpoint_hash:
            mismatches.append("checkpoint_hash mismatch")
        if manifest.get("checkpoint_hash_type") != "sha256":
            mismatches.append("checkpoint_hash_type mismatch")
    if label_mapping is not None and "label_mapping_sha256" in manifest:
        label_hash = _file_sha256(label_mapping)
        if manifest.get("label_mapping_sha256") != label_hash:
            mismatches.append("label_mapping_sha256 mismatch")
    if expected_sample_order_hash is not None and manifest.get("sample_order_hash") != expected_sample_order_hash:
        mismatches.append("sample_order_hash mismatch")
    if verify_shards:
        mismatches.extend(_verify_manifest_shards(artifact_path, manifest, verify_sha256=True))
    if mismatches:
        preview = "; ".join(mismatches[:8])
        raise RuntimeError(f"existing artifact is not reusable: {artifact_path}; {preview}; pass --force to rebuild")
    print(f"[resume] reuse existing artifact: {artifact_path}")
    return {payload_key: payload, path_key: artifact_path, "manifest_path": _sidecar_manifest_path(artifact_path)}


def _existing_token_trace_cache_if_valid(
    *,
    output_dir: Path,
    eval_csv: Path,
    checkpoint_path: str | Path | None,
) -> Mapping[str, Any] | None:
    manifest_path = output_dir / "fp32_token_trace_cache.manifest.json"
    index_path = output_dir / "fp32_token_trace_cache_index.jsonl"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"token trace cache manifest must be a mapping: {manifest_path}")
    mismatches: list[str] = []
    if manifest.get("cache_type") != "fp32_token_trace_cache":
        mismatches.append("cache_type mismatch")
    if manifest.get("stage") != "optional_secondary_token_trace_cache":
        mismatches.append("stage mismatch")
    if manifest.get("eval_csv_sha256") != _file_sha256(eval_csv):
        mismatches.append("eval_csv_sha256 mismatch")
    checkpoint_hash = _file_sha256(checkpoint_path)
    if checkpoint_hash is not None and manifest.get("checkpoint_sha256") != checkpoint_hash:
        mismatches.append("checkpoint_sha256 mismatch")
    if not index_path.exists():
        mismatches.append("missing index jsonl")
    for section_name in ("block_token_trace", "ffn_activation_trace"):
        section = manifest.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for shard_rel in section.get("shards", []) or []:
            shard_path = output_dir / str(shard_rel)
            if not shard_path.exists():
                mismatches.append(f"missing {section_name} shard: {shard_rel}")
    if mismatches:
        preview = "; ".join(mismatches[:8])
        raise RuntimeError(f"existing token trace cache is not reusable: {output_dir}; {preview}; pass --force to rebuild")
    print(f"[resume] reuse existing artifact: {output_dir}")
    return {"manifest": dict(manifest), "output_dir": output_dir, "manifest_path": manifest_path, "index_path": index_path}


def _shard_inventory_for_bundle(manifest: Mapping[str, Any]) -> dict[str, Any]:
    paths = _manifest_shard_paths(manifest)
    hashes = _manifest_shard_sha256(manifest)
    return {
        "num_shards": int(manifest.get("num_shards", len(paths))) if paths or manifest.get("num_shards") is not None else None,
        "shard_paths": paths,
        "shard_sha256": hashes,
    }

def _cfg_to_mapping(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, Mapping):
        return dict(cfg)
    result: dict[str, Any] = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        try:
            result[name] = getattr(cfg, name)
        except AttributeError:
            continue
    return result


def _merged_section(base: Mapping[str, Any], section: str, updates: Mapping[str, Any]) -> dict[str, Any]:
    current = base.get(section, {})
    merged = dict(current) if isinstance(current, Mapping) else {}
    merged.update(updates)
    return merged


def _runtime_cfg(
    cfg: Any,
    *,
    eval_csv: Path,
    label_mapping: Path | None,
    oer_oracle_path: Path,
    check_oer_oracle: bool,
) -> dict[str, Any]:
    """Build a runtime cfg that forces deterministic eval/audio settings."""

    base = _cfg_to_mapping(cfg)
    common_updates: dict[str, Any] = {
        "EVAL_CSV": eval_csv,
        "AUDIO_MODE": "eval",
    }
    if label_mapping is not None:
        common_updates["LABEL_MAPPING"] = label_mapping
    base["FP32_CORE_CACHE"] = _merged_section(base, "FP32_CORE_CACHE", common_updates)
    oer_updates = dict(common_updates)
    oer_updates["ACTIVATION_POSITIONS"] = ("input", "output")
    base["FP32_OER_ORACLE"] = _merged_section(base, "FP32_OER_ORACLE", oer_updates)
    base["FP32_TOKEN_TRACE_CACHE"] = _merged_section(base, "FP32_TOKEN_TRACE_CACHE", common_updates)
    base["FP32_TOKEN_TRACE_ORACLE"] = _merged_section(base, "FP32_TOKEN_TRACE_ORACLE", common_updates)
    sanity_updates = {
        "EVAL_CSV": eval_csv,
        "OER_ORACLE_PATH": oer_oracle_path,
        "CHECK_OER_ORACLE_CACHE": bool(check_oer_oracle),
    }
    base["FP32_GEOMETRY_SANITY"] = _merged_section(base, "FP32_GEOMETRY_SANITY", sanity_updates)
    return base


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_temp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(path)
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path


def _write_sanity_csv(report: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "strict",
        "issue_group",
        "severity",
        "code",
        "message",
        "details_json",
    ]
    rows: list[dict[str, Any]] = []
    for issue_group in ("hard_errors", "blocking_warnings", "warnings"):
        for issue in report.get(issue_group, []) or []:
            if not isinstance(issue, Mapping):
                continue
            rows.append(
                {
                    "status": report.get("status", ""),
                    "strict": bool(report.get("strict", False)),
                    "issue_group": issue_group,
                    "severity": issue.get("severity", ""),
                    "code": issue.get("code", ""),
                    "message": issue.get("message", ""),
                    "details_json": json.dumps(_jsonable(issue.get("details", {})), ensure_ascii=False, sort_keys=True),
                }
            )
    if not rows:
        rows.append(
            {
                "status": report.get("status", "pass"),
                "strict": bool(report.get("strict", False)),
                "issue_group": "none",
                "severity": "info",
                "code": "pass",
                "message": "FP32 reference sanity check passed",
                "details_json": "{}",
            }
        )
    tmp = _atomic_temp_path(path)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path


def _geometry_from_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    sanity = summary.get("sanity", {})
    if not isinstance(sanity, Mapping):
        return {}
    sanity_summary = sanity.get("summary", {})
    if not isinstance(sanity_summary, Mapping):
        return {}
    geometry = sanity_summary.get("geometry", {})
    return geometry if isinstance(geometry, Mapping) else {}


def _flatten_geometry_rows(prefix: str, value: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key) if not prefix else f"{prefix}.{key}"
            _flatten_geometry_rows(name, item, rows)
        return
    rows.append(
        {
            "metric": prefix,
            "value": json.dumps(_jsonable(value), ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else value,
        }
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sanity_group_protocol(sanity_report: Mapping[str, Any], core_manifest: Mapping[str, Any]) -> dict[str, Any]:
    sanity_summary = _mapping_or_empty(sanity_report.get("summary", {}))
    cache_summary = _mapping_or_empty(sanity_summary.get("cache", {}))
    geometry_summary = _mapping_or_empty(sanity_summary.get("geometry", {}))
    protocol_checks = _mapping_or_empty(cache_summary.get("group_protocol", {}))
    sample_counts = (
        cache_summary.get("sample_count_by_analysis_group")
        or geometry_summary.get("sample_count_by_analysis_group")
        or cache_summary.get("analysis_group_counts")
        or geometry_summary.get("analysis_group_counts")
        or core_manifest.get("analysis_group_counts")
    )
    class_counts = (
        cache_summary.get("class_count_by_analysis_group")
        or geometry_summary.get("class_count_by_analysis_group")
        or geometry_summary.get("fisher_effective_class_count_by_analysis_group")
    )
    bucket_counts = cache_summary.get("bucket_counts") or core_manifest.get("bucket_counts")
    return {
        "group_protocol_version": core_manifest.get("group_protocol_version"),
        "bucket_counts": bucket_counts,
        "analysis_group_counts": sample_counts,
        "sample_count_by_analysis_group": sample_counts,
        "class_count_by_analysis_group": class_counts,
        "tail_definition": cache_summary.get("tail_definition") or core_manifest.get("tail_definition"),
        "checks": dict(protocol_checks),
    }

def _write_geometry_artifacts(summary: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> tuple[Path, Path]:
    geometry = _geometry_from_summary(summary)
    payload = {
        "status": summary.get("status", ""),
        "stage": summary.get("stage", ""),
        "geometry": geometry,
        "group_protocol": summary.get("group_protocol", {}),
        "artifacts": summary.get("artifacts", {}),
    }
    _write_json(json_path, payload)

    rows: list[dict[str, Any]] = []
    _flatten_geometry_rows("", geometry, rows)
    if not rows:
        rows.append({"metric": "geometry", "value": "missing"})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(csv_path)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)
    return json_path, csv_path

def _assert_oer_oracle_was_checked(sanity_report: Mapping[str, Any], *, oer_oracle_path: Path) -> None:
    """Require geometry sanity to validate the OER oracle cache."""

    summary = sanity_report.get("summary", {})
    if not isinstance(summary, Mapping):
        raise RuntimeError("sanity report missing summary; cannot confirm OER oracle cache check")
    oracle_summary = summary.get("oer_oracle_cache")
    if not isinstance(oracle_summary, Mapping):
        raise RuntimeError("sanity report missing summary.oer_oracle_cache; entrypoint cannot confirm OER oracle integrity")
    if not bool(oracle_summary.get("checked", False)):
        raise RuntimeError(f"sanity check did not validate OER oracle cache: {oer_oracle_path}")
    if str(oracle_summary.get("status", "")) != "checked":
        raise RuntimeError(f"OER oracle cache status is invalid: status={oracle_summary.get('status')!r}, path={oracle_summary.get('path')!r}")
    row_count = int(oracle_summary.get("row_count", 0))
    expected_row_count = int(oracle_summary.get("expected_row_count", 0))
    if row_count <= 0 or expected_row_count <= 0 or row_count != expected_row_count:
        raise RuntimeError(f"OER oracle row_count mismatch: row_count={row_count}, expected_row_count={expected_row_count}")


def _build_summary(
    *,
    status: str,
    eval_csv: Path,
    label_mapping: Path | None,
    checkpoint_path: str | None,
    core_cache_path: Path,
    oer_oracle_path: Path,
    token_trace_dir: Path | None,
    token_trace_oracle_path: Path | None,
    bundle_path: Path | None,
    sanity_csv_path: Path,
    sanity_raw_report_path: Path,
    sanity_summary_path: Path,
    core_result: Mapping[str, Any] | None,
    oer_result: Mapping[str, Any] | None,
    token_trace_result: Mapping[str, Any] | None,
    token_trace_oracle_result: Mapping[str, Any] | None,
    sanity_report: Mapping[str, Any],
) -> dict[str, Any]:
    core_manifest: Mapping[str, Any] = {}
    if isinstance(core_result, Mapping):
        core_cache = core_result.get("cache", {})
        if isinstance(core_cache, Mapping):
            manifest = core_cache.get("manifest", {})
            if isinstance(manifest, Mapping):
                core_manifest = manifest

    fingerprints = {
        "eval_csv_sha256": core_manifest.get("eval_csv_sha256") or _file_sha256(eval_csv),
        "label_mapping_sha256": core_manifest.get("label_mapping_sha256") or _file_sha256(label_mapping),
        "checkpoint_sha256": core_manifest.get("checkpoint_sha256") or _file_sha256(checkpoint_path),
        "sample_order_hash": core_manifest.get("sample_order_hash"),
    }
    group_protocol = _sanity_group_protocol(sanity_report, core_manifest)
    return {
        "status": status,
        "stage": "fp32_reference_cache_eval_only",
        "summary_type": "fp32_reference_build_summary",
        "eval_csv": eval_csv,
        "label_mapping": label_mapping,
        "checkpoint_path": checkpoint_path,
        "fingerprints": fingerprints,
        "group_protocol": group_protocol,
        "artifacts": {
            "core_eval": core_cache_path,
            "oer_oracle_eval": oer_oracle_path,
            "token_trace_eval": token_trace_dir,
            "token_trace_oracle_eval": token_trace_oracle_path,
            "fp32_reference_bundle": bundle_path,
            "sanity_eval_csv": sanity_csv_path,
            "sanity_raw_report_json": sanity_raw_report_path,
            "sanity_summary_json": sanity_summary_path,
        },
        "core_cache": None if core_result is None else {k: v for k, v in core_result.items() if k != "cache"},
        "oer_oracle": None if oer_result is None else {k: v for k, v in oer_result.items() if k != "oracle"},
        "token_trace_cache": token_trace_result,
        "token_trace_oracle": None if token_trace_oracle_result is None else {k: v for k, v in token_trace_oracle_result.items() if k != "oracle"},
        "sanity": sanity_report,
        "notes": {
            "uses_eval_csv_only": True,
            "uses_unified_audio_input": True,
            "audio_mode": "eval",
            "does_not_train": True,
            "does_not_build_kd_train_cache": True,
            "does_not_calibrate_int4": True,
            "does_not_run_quant_injection": True,
            "random_crop": False,
            "sanity_summary_json_is_total_summary": True,
            "raw_sanity_report_is_nested_under_sanity": True,
        },
    }



def _build_reference_bundle(
    *,
    cfg: Any,
    cache_root: Path,
    core_cache_path: Path,
    oer_oracle_path: Path,
    token_trace_oracle_path: Path | None,
    core_result: Mapping[str, Any] | None,
    oer_result: Mapping[str, Any] | None,
    token_trace_oracle_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    core_manifest = _manifest_from_result(core_result, "cache")
    oer_manifest = _manifest_from_result(oer_result, "oracle")
    token_manifest = _manifest_from_result(token_trace_oracle_result, "oracle")
    checkpoint_hash = core_manifest.get("checkpoint_hash") or core_manifest.get("checkpoint_sha256") or oer_manifest.get("checkpoint_hash") or oer_manifest.get("checkpoint_sha256") or _checkpoint_sha256_for_manifest(cfg)
    eval_csv_hash = core_manifest.get("eval_csv_sha256") or oer_manifest.get("eval_csv_sha256") or _file_sha256(core_manifest.get("eval_csv"))
    label_mapping_hash = core_manifest.get("label_mapping_sha256") or _file_sha256(core_manifest.get("label_mapping"))
    group_protocol_version = core_manifest.get("group_protocol_version", 2)
    token_available = bool(token_trace_oracle_path is not None and token_trace_oracle_result is not None)
    core_shards = _shard_inventory_for_bundle(core_manifest)
    oer_shards = _shard_inventory_for_bundle(oer_manifest)
    token_shards = _shard_inventory_for_bundle(token_manifest) if token_available else {"num_shards": None, "shard_paths": [], "shard_sha256": {}}
    return {
        "bundle_name": "fp32_reference_eval",
        "dataset_name": core_manifest.get("dataset_name") or oer_manifest.get("dataset_name") or _dataset_name(cfg),
        "split": "eval",
        "core_cache": {
            "path": _relative_artifact_path(core_cache_path, cache_root),
            "identity": _file_identity(core_cache_path),
            "storage_format": core_manifest.get("storage_format"),
            "checkpoint_sha256": core_manifest.get("checkpoint_sha256"),
            "checkpoint_hash": core_manifest.get("checkpoint_hash"),
            "checkpoint_hash_type": core_manifest.get("checkpoint_hash_type"),
            "layout_protocol_version": core_manifest.get("layout_protocol_version"),
            "embedding_dim": core_manifest.get("embedding_dim"),
            "embedding_pooling": core_manifest.get("embedding_pooling"),
            "sample_order_hash": core_manifest.get("sample_order_hash"),
            "num_samples": core_manifest.get("num_samples"),
            "num_shards": core_shards.get("num_shards"),
            "shard_paths": core_shards.get("shard_paths"),
            "shard_sha256": core_shards.get("shard_sha256"),
        },
        "oer_oracle_cache": {
            "path": _relative_artifact_path(oer_oracle_path, cache_root),
            "identity": _file_identity(oer_oracle_path),
            "storage_format": oer_manifest.get("storage_format"),
            "target_count": len(oer_manifest.get("target_modules") or []),
            "topk_unit": oer_manifest.get("topk_unit"),
            "topk_source": oer_manifest.get("topk_source"),
            "topk_ratio": oer_manifest.get("topk_ratio"),
            "topk_k_policy": oer_manifest.get("topk_k_policy"),
            "valid_unit": oer_manifest.get("valid_unit"),
            "valid_unit_count_field": oer_manifest.get("valid_unit_count_field"),
            "padding_excluded": oer_manifest.get("padding_excluded"),
            "layout_protocol_version": oer_manifest.get("layout_protocol_version"),
            "activation_positions": oer_manifest.get("activation_positions"),
            "contains_fp32_oracle_energy": bool(oer_manifest.get("contains_fp32_oracle_energy", False)),
            "contains_fp32_total_energy": bool(oer_manifest.get("contains_fp32_total_energy", False)),
            "energy_scope": oer_manifest.get("energy_scope"),
            "num_shards": oer_shards.get("num_shards"),
            "shard_paths": oer_shards.get("shard_paths"),
            "shard_sha256": oer_shards.get("shard_sha256"),
        },
        "optional_token_trace_oracle_cache": {
            "path": _relative_artifact_path(token_trace_oracle_path, cache_root) if token_trace_oracle_path is not None else None,
            "identity": _file_identity(token_trace_oracle_path) if token_available else None,
            "available": token_available,
            "storage_format": token_manifest.get("storage_format") if token_available else None,
            "required_for": "block_output_token_survival",
            "row_count": token_manifest.get("row_count") if token_available else None,
            "layer_count": token_manifest.get("num_layers") if token_available else None,
            "topk_unit": token_manifest.get("topk_unit") if token_available else "token",
            "topk_source": token_manifest.get("topk_source") if token_available else "fp32_token_energy",
            "topk_ratio": token_manifest.get("topk_ratio") if token_available else None,
            "topk_k_policy": token_manifest.get("topk_k_policy") if token_available else None,
            "valid_unit": token_manifest.get("valid_unit") if token_available else None,
            "valid_unit_count_field": token_manifest.get("valid_unit_count_field") if token_available else None,
            "padding_excluded": token_manifest.get("padding_excluded") if token_available else None,
            "layout_protocol_version": token_manifest.get("layout_protocol_version") if token_available else None,
            "num_shards": token_shards.get("num_shards") if token_available else None,
            "shard_paths": token_shards.get("shard_paths") if token_available else [],
            "shard_sha256": token_shards.get("shard_sha256") if token_available else {},
        },
        "shared_identity": {
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_hash_type": core_manifest.get("checkpoint_hash_type") or "sha256",
            "eval_csv_hash": eval_csv_hash,
            "label_mapping_hash": label_mapping_hash,
            "audio_config_hash": _audio_config_hash(cfg),
            "group_protocol_version": group_protocol_version,
            "layout_protocol_version": core_manifest.get("layout_protocol_version"),
        },
    }

def build_fp32_reference_cache(
    cfg: Any | None = None,
    *,
    include_token_trace: bool | None = None,
    force: bool | None = None,
) -> dict[str, Any]:
    """Build FP32 reference caches for quant injection experiments."""

    cfg = _default_cfg() if cfg is None else cfg
    eval_csv, label_mapping, checkpoint_path = _preflight_inputs(cfg)
    cache_root = _cache_root(cfg)
    reports_root = _reports_root(cfg)
    core_cache_path = cache_root / CORE_CACHE_FILENAME
    oer_oracle_path = cache_root / OER_ORACLE_FILENAME
    token_trace_dir = cache_root / TOKEN_TRACE_DIRNAME
    token_trace_oracle_path = cache_root / TOKEN_TRACE_ORACLE_FILENAME
    bundle_path = cache_root / BUNDLE_FILENAME
    sanity_csv_path = reports_root / SANITY_CSV_FILENAME
    sanity_raw_report_path = reports_root / SANITY_RAW_REPORT_FILENAME
    sanity_summary_path = reports_root / SANITY_SUMMARY_FILENAME
    geometry_json_path = reports_root / GEOMETRY_JSON_FILENAME
    geometry_csv_path = reports_root / GEOMETRY_CSV_FILENAME
    include_trace = _include_token_trace(cfg) if include_token_trace is None else bool(include_token_trace)
    include_trace_oracle = _include_token_trace_oracle(cfg)
    force_rebuild = _force(cfg) if force is None else bool(force)

    runtime_cfg = _runtime_cfg(
        cfg,
        eval_csv=eval_csv,
        label_mapping=label_mapping,
        oer_oracle_path=oer_oracle_path,
        check_oer_oracle=True,
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    model: nn.Module = _restore_fp32_teacher(runtime_cfg)
    model.eval()
    eval_loader: DataLoader = _build_eval_loader(runtime_cfg)

    core_result: Mapping[str, Any] | None = None
    oer_result: Mapping[str, Any] | None = None
    token_trace_result: Mapping[str, Any] | None = None
    token_trace_oracle_result: Mapping[str, Any] | None = None
    sanity_report: Mapping[str, Any] = {}

    try:
        core_result = None if force_rebuild else _ensure_existing_artifact_valid(
            artifact_path=core_cache_path,
            payload_key="cache",
            path_key="cache_path",
            expected_manifest={
                "cache_type": "fp32_core_cache",
                "stage": "quant_injection_pre_fp32_reference",
                "storage_format": "sample_sharded_v1",
                "checkpoint_hash_type": "sha256",
                "layout_protocol_version": 1,
            },
            eval_csv=eval_csv,
            checkpoint_path=checkpoint_path,
            label_mapping=label_mapping,
        )
        if core_result is None:
            core_result = export_fp32_core_cache(
                runtime_cfg,
                output_path=core_cache_path,
                force=force_rebuild,
                model=model,
                eval_loader=eval_loader,
            )

        core_manifest = _manifest_from_result(core_result, "cache")
        sample_order_hash = None if core_manifest.get("sample_order_hash") is None else str(core_manifest.get("sample_order_hash"))
        oer_result = None if force_rebuild else _ensure_existing_artifact_valid(
            artifact_path=oer_oracle_path,
            payload_key="oracle",
            path_key="oracle_path",
            expected_manifest={
                "cache_type": "fp32_oer_oracle",
                "oracle_source": "fp32_real_activation",
                "stage": "quant_injection_pre_fp32_oer_oracle",
                "storage_format": "layer_sample_sharded_v1",
                "topk_unit": "scalar",
                "topk_source": "fp32_scalar_energy",
                "topk_k_policy": "ceil(topk_ratio * valid_unit_count)",
                "valid_unit": "valid_scalar",
                "valid_unit_count_field": "valid_scalar_count",
                "padding_excluded": True,
                "layout_protocol_version": 1,
            },
            eval_csv=eval_csv,
            checkpoint_path=checkpoint_path,
            expected_sample_order_hash=sample_order_hash,
        )
        if oer_result is None:
            oer_result = export_fp32_oer_oracle(
                runtime_cfg,
                output_path=oer_oracle_path,
                force=force_rebuild,
                model=model,
                eval_loader=eval_loader,
            )

        if include_trace_oracle:
            token_trace_oracle_result = None if force_rebuild else _ensure_existing_artifact_valid(
                artifact_path=token_trace_oracle_path,
                payload_key="oracle",
                path_key="oracle_path",
                expected_manifest={
                    "cache_type": "fp32_token_trace_oracle",
                    "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
                    "oracle_source": "fp32_multi_position_token_energy",
                    "stage": "quant_injection_pre_fp32_token_trace_oracle",
                    "storage_format": ORACLE_STORAGE_FORMAT,
                    "positions": ["block_output", "input_activation"],
                    "available_positions": ["block_output", "input_activation"],
                    "topk_unit": "token",
                    "topk_source": "fp32_token_energy",
                    "topk_k_policy": "ceil(topk_ratio * valid_unit_count)",
                    "valid_unit": "valid_token",
                    "valid_unit_count_field": "valid_token_count",
                    "padding_excluded": True,
                    "layout_protocol_version": 1,
                },
                eval_csv=eval_csv,
                checkpoint_path=checkpoint_path,
                expected_sample_order_hash=sample_order_hash,
            )
            if token_trace_oracle_result is None:
                token_trace_oracle_result = export_fp32_token_trace_oracle(
                    runtime_cfg,
                    output_path=token_trace_oracle_path,
                    force=force_rebuild,
                    model=model,
                    eval_loader=eval_loader,
                )
        if include_trace:
            token_trace_result = None if force_rebuild else _existing_token_trace_cache_if_valid(
                output_dir=token_trace_dir,
                eval_csv=eval_csv,
                checkpoint_path=checkpoint_path,
            )
            if token_trace_result is None:
                token_trace_result = export_fp32_token_trace_cache(
                    runtime_cfg,
                    output_dir=token_trace_dir,
                    force=force_rebuild,
                    model=model,
                    eval_loader=eval_loader,
                )
        try:
            sanity_report = run_fp32_geometry_sanity_check(
                runtime_cfg,
                core_cache_path=core_cache_path,
                report_path=sanity_raw_report_path,
                model=model,
                strict=_strict_sanity(runtime_cfg),
                save=True,
            )
            _assert_oer_oracle_was_checked(sanity_report, oer_oracle_path=oer_oracle_path)
            status = str(sanity_report.get("status", "pass"))
        except FP32GeometrySanityError as exc:
            sanity_report = exc.report
            status = str(sanity_report.get("status", "hard_error"))
            _write_json(sanity_raw_report_path, sanity_report)
            _write_sanity_csv(sanity_report, sanity_csv_path)
            summary = _build_summary(
                status=status,
                eval_csv=eval_csv,
                label_mapping=label_mapping,
                checkpoint_path=checkpoint_path,
                core_cache_path=core_cache_path,
                oer_oracle_path=oer_oracle_path,
                token_trace_dir=token_trace_dir if include_trace else None,
                token_trace_oracle_path=token_trace_oracle_path if include_trace_oracle else None,
                bundle_path=None,
                sanity_csv_path=sanity_csv_path,
                sanity_raw_report_path=sanity_raw_report_path,
                sanity_summary_path=sanity_summary_path,
                core_result=core_result,
                oer_result=oer_result,
                token_trace_result=token_trace_result,
                token_trace_oracle_result=token_trace_oracle_result,
                sanity_report=sanity_report,
            )
            summary["artifacts"]["geometry_json"] = geometry_json_path
            summary["artifacts"]["geometry_csv"] = geometry_csv_path
            _write_geometry_artifacts(summary, json_path=geometry_json_path, csv_path=geometry_csv_path)
            _write_json(sanity_summary_path, summary)
            raise

        _write_sanity_csv(sanity_report, sanity_csv_path)
        summary = _build_summary(
            status=status,
            eval_csv=eval_csv,
            label_mapping=label_mapping,
            checkpoint_path=checkpoint_path,
            core_cache_path=core_cache_path,
            oer_oracle_path=oer_oracle_path,
            token_trace_dir=token_trace_dir if include_trace else None,
            token_trace_oracle_path=token_trace_oracle_path if include_trace_oracle else None,
            bundle_path=bundle_path,
            sanity_csv_path=sanity_csv_path,
            sanity_raw_report_path=sanity_raw_report_path,
            sanity_summary_path=sanity_summary_path,
            core_result=core_result,
            oer_result=oer_result,
            token_trace_result=token_trace_result,
            token_trace_oracle_result=token_trace_oracle_result,
            sanity_report=sanity_report,
        )
        summary["artifacts"]["geometry_json"] = geometry_json_path
        summary["artifacts"]["geometry_csv"] = geometry_csv_path
        _write_geometry_artifacts(summary, json_path=geometry_json_path, csv_path=geometry_csv_path)
        bundle = _build_reference_bundle(
            cfg=runtime_cfg,
            cache_root=cache_root,
            core_cache_path=core_cache_path,
            oer_oracle_path=oer_oracle_path,
            token_trace_oracle_path=token_trace_oracle_path if include_trace_oracle else None,
            core_result=core_result,
            oer_result=oer_result,
            token_trace_oracle_result=token_trace_oracle_result,
        )
        _write_json(bundle_path, bundle)
        _write_json(sanity_summary_path, summary)
        return summary
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(cfg: Any | None = None) -> dict[str, Any]:
    return build_fp32_reference_cache(cfg)


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.6g}"


def _print_cli_summary(summary: Mapping[str, Any]) -> None:
    artifacts = summary.get("artifacts", {})
    geometry = _geometry_from_summary(summary)
    group_protocol = _mapping_or_empty(summary.get("group_protocol", {}))
    print(f"FP32 reference cache status: {summary.get('status', '')}")
    if isinstance(artifacts, Mapping):
        for key in (
            "core_eval",
            "oer_oracle_eval",
            "token_trace_oracle_eval",
            "fp32_reference_bundle",
            "sanity_eval_csv",
            "sanity_raw_report_json",
            "sanity_summary_json",
            "geometry_json",
            "geometry_csv",
        ):
            value = artifacts.get(key)
            if value is not None:
                print(f"{key}: {value}")

    if not geometry:
        print("geometry: missing")
        if group_protocol:
            print(f"  sample_count_by_analysis_group: {group_protocol.get('sample_count_by_analysis_group', group_protocol.get('analysis_group_counts', {}))}")
            print(f"  class_count_by_analysis_group: {group_protocol.get('class_count_by_analysis_group', {})}")
        return

    margin = geometry.get("margin", {})
    trace = geometry.get("class_trace_variance", {})
    print("geometry:")
    print(f"  fisher_global_anchored: {_format_float(geometry.get('fisher_global_anchored', geometry.get('fisher_ratio')))}")
    fisher_by_group = geometry.get("fisher_by_analysis_group", {})
    if isinstance(fisher_by_group, Mapping):
        print(f"  fisher_by_analysis_group: {fisher_by_group}")
    if isinstance(margin, Mapping):
        print(f"  margin.positive_ratio: {_format_float(margin.get('positive_ratio'))}")
        print(f"  margin.mean: {_format_float(margin.get('mean'))}")
        print(f"  margin.median: {_format_float(margin.get('median'))}")
    if isinstance(trace, Mapping):
        print(f"  class_trace_variance.mean: {_format_float(trace.get('mean'))}")
        print(f"  class_trace_variance.p95: {_format_float(trace.get('p95'))}")
    sample_counts = geometry.get("sample_count_by_analysis_group") or group_protocol.get("sample_count_by_analysis_group") or geometry.get("analysis_group_counts", geometry.get("shot_group_counts", {}))
    class_counts = geometry.get("class_count_by_analysis_group") or group_protocol.get("class_count_by_analysis_group", {})
    print(f"  sample_count_by_analysis_group: {sample_counts}")
    print(f"  class_count_by_analysis_group: {class_counts}")
    checks = _mapping_or_empty(group_protocol.get("checks", {}))
    if checks:
        print(f"  group_protocol_checks: {checks}")


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build eval-only FP32 reference cache and geometry sanity results.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing cache artifacts.")
    parser.add_argument("--include-token-trace", action="store_true", help="Also build token trace cache.")
    args = parser.parse_args()

    try:
        summary = build_fp32_reference_cache(
            None,
            force=True if args.force else None,
            include_token_trace=True if args.include_token_trace else None,
        )
    except FP32GeometrySanityError as exc:
        print("FP32 reference cache geometry sanity check failed.")
        report = exc.report if isinstance(exc.report, Mapping) else {}
        sanity_summary = report.get("summary", {}) if isinstance(report, Mapping) else {}
        geometry = {}
        if isinstance(sanity_summary, Mapping):
            raw_geometry = sanity_summary.get("geometry", {})
            geometry = raw_geometry if isinstance(raw_geometry, Mapping) else {}
        if geometry:
            print(json.dumps(_jsonable(geometry), ensure_ascii=False, indent=2))
        raise

    _print_cli_summary(summary)
    return 0

__all__ = [
    "BUNDLE_FILENAME",
    "CORE_CACHE_FILENAME",
    "GEOMETRY_CSV_FILENAME",
    "GEOMETRY_JSON_FILENAME",
    "OER_ORACLE_FILENAME",
    "SANITY_CSV_FILENAME",
    "SANITY_RAW_REPORT_FILENAME",
    "SANITY_SUMMARY_FILENAME",
    "TOKEN_TRACE_DIRNAME",
    "TOKEN_TRACE_ORACLE_FILENAME",
    "build_fp32_reference_cache",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(_cli())
