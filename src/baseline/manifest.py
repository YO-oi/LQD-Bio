from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HASH_EXCLUDE_KEYS = {"run_manifest_hash", "run_manifest_hash_algorithm", "manifest_path"}


def path_str(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).resolve(strict=False))


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in manifest.items() if str(key) not in _HASH_EXCLUDE_KEYS}


def attach_run_manifest_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(manifest))
    manifest_hash = canonical_json_hash(_manifest_hash_payload(payload))
    payload["run_manifest_hash"] = manifest_hash
    payload["run_manifest_hash_algorithm"] = "sha256-canonical-json-before-run_manifest_hash"
    return payload


def write_run_manifest(output_dir: str | Path, manifest: Mapping[str, Any], *, filename: str = "run_manifest.json") -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = attach_run_manifest_hash(manifest)
    manifest_path = directory / filename
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest_path


def write_run_manifests(output_dirs: Mapping[str, str | Path], manifest: Mapping[str, Any]) -> dict[str, str]:
    written: dict[str, str] = {}
    for name, directory in output_dirs.items():
        written[str(name)] = path_str(write_run_manifest(directory, manifest)) or str(Path(directory) / "run_manifest.json")
    return written


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_commit(project_root: str | Path | None = None, *, required: bool = True) -> str | None:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except Exception as exc:
        if required:
            raise RuntimeError(f"git commit is required for formal Baseline manifest, but git rev-parse failed under {root}") from exc
        return None
    value = result.stdout.strip()
    if not value and required:
        raise RuntimeError(f"git commit is required for formal Baseline manifest, but git rev-parse returned empty output under {root}")
    return value or None

def hash_existing_paths(paths: Mapping[str, str | Path | None]) -> dict[str, str | None]:
    return {str(name): file_sha256(path) for name, path in paths.items()}


def split_hashes(
    *,
    train_csv: str | Path | None,
    eval_csv: str | Path | None,
    test_csv: str | Path | None,
    calibration_csv: str | Path | None = None,
) -> dict[str, str | None]:
    hashes = {
        "train": file_sha256(train_csv),
        "eval": file_sha256(eval_csv),
        "test": file_sha256(test_csv),
    }
    if calibration_csv is not None:
        hashes["calibration"] = file_sha256(calibration_csv)
    return hashes


def manifest_hash_matches(manifest_path: str | Path, expected_hash: str | None) -> bool:
    """Deprecated: use artifact_manifest_matches() for manifest/content audits."""
    warnings.warn(
        "manifest_hash_matches() is deprecated; use artifact_manifest_matches() so stored and recomputed hashes are both audited",
        DeprecationWarning,
        stacklevel=2,
    )
    if not expected_hash:
        return False
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        return False
    stored = str(manifest.get("run_manifest_hash", ""))
    recomputed = canonical_json_hash(_manifest_hash_payload(manifest))
    return stored == str(expected_hash) and recomputed == str(expected_hash)

def artifact_manifest_matches(
    manifest_path: str | Path,
    expected: Mapping[str, Any],
    *,
    required_keys: Sequence[str] | None = None,
) -> tuple[bool, list[str]]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        return False, ["manifest is not a JSON object"]
    keys = list(required_keys) if required_keys is not None else sorted(str(key) for key in expected)
    mismatches: list[str] = []
    for key in keys:
        if manifest.get(key) != expected.get(key):
            mismatches.append(f"{key}: existing={manifest.get(key)!r} expected={expected.get(key)!r}")
    stored_hash = manifest.get("run_manifest_hash")
    if stored_hash:
        recomputed = canonical_json_hash(_manifest_hash_payload(manifest))
        if str(stored_hash) != recomputed:
            mismatches.append(f"run_manifest_hash: existing={stored_hash!r} recomputed={recomputed!r}")
    else:
        mismatches.append("run_manifest_hash missing")
    return not mismatches, mismatches


def qparams_reuse_allowed(
    manifest_path: str | Path,
    expected_manifest: Mapping[str, Any],
    *,
    qparams_csv_path: str | Path | None = None,
    qparams_hash_value: str | None = None,
) -> tuple[bool, list[str]]:
    required = (
        "stage",
        "teacher_checkpoint_sha256",
        "split_hashes",
        "split_hash",
        "label_mapping_hash",
        "calibration_hash",
        "quantized_layers_hash",
        "bitwidth",
        "pooling",
        "input_length_samples",
        "seed",
        "audio_input_py_sha256",
        "qparams_csv",
        "qparams_csv_hash",
        "qparams_csv_sha256",
        "qparams_hash",
    )
    ok, mismatches = artifact_manifest_matches(manifest_path, expected_manifest, required_keys=required)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        return False, [*mismatches, "manifest is not a JSON object"]
    csv_path_value = qparams_csv_path or expected_manifest.get("qparams_csv") or manifest.get("qparams_csv")
    current_csv_hash = file_sha256(csv_path_value)
    if not current_csv_hash:
        mismatches.append(f"qparams_csv artifact missing or unreadable: {csv_path_value!r}")
    for field in ("qparams_csv_hash", "qparams_csv_sha256"):
        expected_value = expected_manifest.get(field)
        manifest_value = manifest.get(field)
        if not expected_value:
            mismatches.append(f"{field}: expected manifest is missing current qparams artifact hash")
        if not manifest_value:
            mismatches.append(f"{field}: existing manifest is missing qparams artifact hash")
        if current_csv_hash and expected_value and str(current_csv_hash) != str(expected_value):
            mismatches.append(f"{field}: current={current_csv_hash!r} expected={expected_value!r}")
        if current_csv_hash and manifest_value and str(current_csv_hash) != str(manifest_value):
            mismatches.append(f"{field}: current={current_csv_hash!r} existing={manifest_value!r}")
    if qparams_hash_value is not None:
        for label, value in (("expected", expected_manifest.get("qparams_hash")), ("existing", manifest.get("qparams_hash"))):
            if not value:
                mismatches.append(f"qparams_hash: {label} manifest is missing semantic qparams hash")
            elif str(value) != str(qparams_hash_value):
                mismatches.append(f"qparams_hash: current={qparams_hash_value!r} {label}={value!r}")
    elif not expected_manifest.get("qparams_hash") or not manifest.get("qparams_hash"):
        mismatches.append("qparams_hash missing; pass qparams_hash_value from the current qparams artifact for hard reuse audit")
    return not mismatches, mismatches

def baseline_common_manifest(
    *,
    stage: str,
    dataset_name: str,
    checkpoint: str | Path | None,
    train_csv: str | Path | None,
    eval_csv: str | Path | None,
    test_csv: str | Path | None,
    label_mapping: str | Path | None,
    calibration_csv: str | Path | None = None,
    calibration_manifest: str | Path | None = None,
    calibration_hash: str | None = None,
    quantized_layers: Sequence[Mapping[str, Any]] | None = None,
    bitwidth: str | Mapping[str, Any] = "fp32",
    pooling: str = "mask_aware_mean(token_features_after_last_encoder_layer)",
    input_length_samples: int | None = None,
    input_length_seconds: float | None = None,
    sample_rate: int | None = None,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
    require_git_commit: bool = True,
) -> dict[str, Any]:
    split_hash_map = split_hashes(
        train_csv=train_csv,
        eval_csv=eval_csv,
        test_csv=test_csv,
        calibration_csv=calibration_csv,
    )
    layers = [dict(layer) for layer in (quantized_layers or [])]
    audio_input_py = PROJECT_ROOT / "src" / "data" / "audio_input.py"
    group_utils_py = PROJECT_ROOT / "src" / "utils" / "group_utils.py"
    audio_input_py_sha256 = file_sha256(audio_input_py)
    group_utils_py_sha256 = file_sha256(group_utils_py)
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "stage": str(stage),
        "dataset_name": str(dataset_name),
        "created_at": utc_timestamp(),
        "checkpoint": path_str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "split_csv": {
            "train": path_str(train_csv),
            "eval": path_str(eval_csv),
            "test": path_str(test_csv),
            "calibration": path_str(calibration_csv),
        },
        "split_hashes": split_hash_map,
        "split_hash": canonical_json_hash(split_hash_map),
        "label_mapping": path_str(label_mapping),
        "label_mapping_hash": file_sha256(label_mapping),
        "calibration_csv": path_str(calibration_csv),
        "calibration_manifest": path_str(calibration_manifest),
        "calibration_csv_hash": file_sha256(calibration_csv),
        "calibration_manifest_hash": file_sha256(calibration_manifest),
        "calibration_hash": calibration_hash or file_sha256(calibration_manifest) or file_sha256(calibration_csv),
        "quantized_layers": layers,
        "quantized_layers_hash": canonical_json_hash(layers),
        "bitwidth": bitwidth,
        "pooling": str(pooling),
        "audio_input_py": path_str(audio_input_py),
        "audio_input_py_sha256": audio_input_py_sha256,
        "audio_input_hash": audio_input_py_sha256,
        "group_utils_py": path_str(group_utils_py),
        "group_utils_py_sha256": group_utils_py_sha256,
        "group_utils_hash": group_utils_py_sha256,
        "input_length_samples": None if input_length_samples is None else int(input_length_samples),
        "input_length_seconds": None if input_length_seconds is None else float(input_length_seconds),
        "sample_rate": None if sample_rate is None else int(sample_rate),
        "seed": None if seed is None else int(seed),
        "git_commit": git_commit(required=bool(require_git_commit)),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


__all__ = [
    "artifact_manifest_matches",
    "attach_run_manifest_hash",
    "baseline_common_manifest",
    "canonical_json_dumps",
    "canonical_json_hash",
    "file_sha256",
    "git_commit",
    "hash_existing_paths",
    "load_json",
    "path_str",
    "qparams_reuse_allowed",
    "split_hashes",
    "utc_timestamp",
    "write_run_manifest",
    "write_run_manifests",
]
