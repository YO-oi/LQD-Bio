from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from src.baseline.manifest import attach_run_manifest_hash, file_sha256, path_str, utc_timestamp
from src.train.train_beats_classifier_head import dataset_paths


STAGE_NAME = "baseline_fp32_teacher_staging"


def default_staged_fp32_teacher_checkpoint(*, seed: int, baseline_root: str | Path) -> Path:
    return Path(baseline_root) / f"seed_{int(seed)}" / "checkpoints" / f"fp32_teacher_seed_{int(seed)}.pt"


def dataset_canonical_fp32_teacher_checkpoint(dataset_name: str | None) -> Path:
    return dataset_paths(dataset_name, None).output_checkpoint


def _same_path(left: Path, right: Path) -> bool:
    return str(left.resolve(strict=False)) == str(right.resolve(strict=False))


def _write_stage_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(attach_run_manifest_hash(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def stage_fp32_teacher_checkpoint(
    *,
    dataset_name: str | None,
    seed: int,
    baseline_root: str | Path,
    explicit_teacher_checkpoint: str | Path | None = None,
    staged_checkpoint: str | Path | None = None,
    force_restage: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage the dataset FP32 teacher into the per-seed baseline checkpoint slot."""
    source = Path(explicit_teacher_checkpoint) if explicit_teacher_checkpoint is not None else dataset_canonical_fp32_teacher_checkpoint(dataset_name)
    target = Path(staged_checkpoint) if staged_checkpoint is not None else default_staged_fp32_teacher_checkpoint(seed=seed, baseline_root=baseline_root)
    manifest_path = target.with_name(f"{target.stem}.staging_manifest.json")
    source_kind = "explicit_teacher_checkpoint" if explicit_teacher_checkpoint is not None else "dataset_canonical_fp32_teacher"

    if not source.exists() or not source.is_file():
        raise FileNotFoundError(
            "FP32 teacher source checkpoint not found for baseline staging: "
            f"{source}. Train it first with src/train/train_beats_classifier_head.py "
            "or pass --teacher-checkpoint."
        )

    source_hash = file_sha256(source)
    if source_hash is None:
        raise RuntimeError(f"cannot hash FP32 teacher source checkpoint: {source}")

    target_hash_before = file_sha256(target)
    if _same_path(source, target):
        action = "source_already_staged"
    elif target.exists():
        if target_hash_before == source_hash:
            action = "reuse_existing_staged_checkpoint"
        elif bool(force_restage):
            action = "overwrite_mismatched_staged_checkpoint"
            if not bool(dry_run):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        else:
            raise FileExistsError(
                "staged FP32 teacher checkpoint already exists but sha256 differs from the requested source: "
                f"staged={target} source={source}. Pass --force-stage-fp32-teacher to replace it."
            )
    else:
        action = "create_staged_checkpoint"
        if not bool(dry_run):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    target_hash_after = source_hash if bool(dry_run) and action in {"create_staged_checkpoint", "overwrite_mismatched_staged_checkpoint"} else file_sha256(target)
    if not bool(dry_run) and target_hash_after != source_hash:
        raise RuntimeError(f"staged FP32 teacher hash mismatch: source={source_hash}, staged={target_hash_after}")

    payload = {
        "stage": STAGE_NAME,
        "dataset_name": dataset_name,
        "seed": int(seed),
        "created_at": utc_timestamp(),
        "source_kind": source_kind,
        "source_checkpoint": path_str(source),
        "source_checkpoint_sha256": source_hash,
        "staged_checkpoint": path_str(target),
        "staged_checkpoint_sha256": target_hash_after,
        "stage_action": action,
        "force_restage": bool(force_restage),
        "dry_run": bool(dry_run),
    }
    if not bool(dry_run):
        _write_stage_manifest(manifest_path, payload)
    payload["stage_manifest"] = path_str(manifest_path) if not bool(dry_run) else None
    return dict(payload)


__all__ = [
    "STAGE_NAME",
    "dataset_canonical_fp32_teacher_checkpoint",
    "default_staged_fp32_teacher_checkpoint",
    "stage_fp32_teacher_checkpoint",
]
