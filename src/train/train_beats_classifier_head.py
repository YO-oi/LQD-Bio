from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from BEATs import BEATs, BEATsConfig
from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
from src.baseline.export_eval import DEFAULT_TOKEN_EXPORT, DEFAULT_TOKEN_LAYERS, run_export_eval
from src.baseline.manifest import canonical_json_hash, file_sha256, git_commit
from src.utils.group_utils import (
    BASE_BUCKETS,
    analysis_group_mask,
    bucket_from_train_label_n,
    read_canonical_bucket,
)


SELECTION_METRICS = ("eval_loss", "eval_acc", "eval_macro_f1_label_universe_nanmean")
DEFAULT_SOURCE_BEATS_FILENAME = "BEATs_iter3_plus_AS2M.pt"

"""
婵☆垪鈧磭鈧兘鏌呮径瀣仴闁哄牃鍋撳ù鍏济ぐ鏌ユ焻婢跺绨板☉?
--selection-metric eval_loss
--selection-metric eval_acc
--selection-metric eval_macro_f1_label_universe_nanmean
Baseline FP32 teacher default: canonical best eval_macro_f1_label_universe_nanmean
"""
@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path
    train_csv: Path
    eval_csv: Path
    test_csv: Path
    full_csv: Path
    label_mapping: Path
    output_checkpoint: Path


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)




def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive finite float, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze BEATs and train a dataset-specific classifier head.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset key from config.DATASETS; default: config.ACTIVE_DATASET.")
    parser.add_argument("--checkpoint", default=None, help="Source pretrained BEATs checkpoint; default: BEATs/BEATs_iter3_plus_AS2M.pt.")
    parser.add_argument("--output", default=None, help="Output dataset FP32 teacher checkpoint; default: <dataset_root>/BEATs_<dataset>.pt.")
    parser.add_argument("--log-dir", default=None, help="Epoch CSV log directory; default: BASE_DIR/outputs/log.")
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=None)
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="eval_macro_f1_label_universe_nanmean", help="Metric used to choose the best checkpoint.")
    parser.add_argument("--skip-export-eval", action="store_true", help="Skip automatic eval/test export after training; useful for debug runs.")
    parser.add_argument("--export-root", default=None, help="Root for automatic export_eval outputs; default: BASE_DIR/outputs.")
    parser.add_argument("--export-name", default="fp32_teacher", help="Subdirectory name for export_eval artifacts.")
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT, help="Token feature export mode for automatic export_eval outputs.")
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS, help="Comma-separated encoder layer indices for token feature export.")
    parser.add_argument("--export-all-token-layers", action="store_true", help="Export all encoder block output token layers instead of --token-layers.")
    parser.add_argument("--max-export-payload-gib", type=float, default=16.0, help="Abort export_eval if estimated tensor payload exceeds this GiB threshold.")
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=int(getattr(project_config, "SPLIT_SEED", 42)))
    parser.add_argument("--device", default=None, help="Device string; default: config.DEVICE.")
    parser.add_argument("--patience", type=_nonnegative_int, default=5, help="Early stop patience; 0 disables early stop.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast for the predictor head only; the frozen BEATs backbone stays FP32.")
    parser.add_argument("--max-train-batches", type=_positive_int, default=None, help="Debug only: limit train batches per epoch.")
    parser.add_argument("--max-eval-batches", type=_positive_int, default=None, help="Debug only: limit eval batches per epoch.")
    parser.add_argument("--allow-missing-git-commit", action="store_true", help="Debug only: allow non-formal checkpoints when git rev-parse HEAD is unavailable.")
    parser.add_argument("--state-output", default=None, help="Training-state checkpoint; default: <output stem>.train_state.pt.")
    parser.add_argument("--resume-state", default=None, help="Resume classifier-head training from a training-state checkpoint.")
    parser.add_argument("--save-state-every-batches", type=_nonnegative_int, default=100, help="Save training state every N train batches; 0 disables mid-epoch saves.")
    parser.add_argument("--no-state-checkpoint", action="store_true", help="Disable training-state checkpointing.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output checkpoint.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    seed_fn = getattr(project_config, "seed_everything", None)
    if callable(seed_fn):
        seed_fn(int(seed))
        return
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def default_source_beats_checkpoint() -> Path:
    beats = getattr(project_config, "BEATS", {})
    if isinstance(beats, Mapping):
        for key in ("SOURCE_CHECKPOINT", "PRETRAINED_CHECKPOINT"):
            value = beats.get(key)
            if value is not None:
                return Path(value)
        model_dir = beats.get("MODEL_DIR")
        if model_dir is not None:
            return Path(model_dir) / DEFAULT_SOURCE_BEATS_FILENAME
    return PROJECT_ROOT / "BEATs" / DEFAULT_SOURCE_BEATS_FILENAME


def resolve_source_checkpoint_path(checkpoint: str | Path | None) -> Path:
    return Path(checkpoint) if checkpoint is not None else default_source_beats_checkpoint()


def dataset_paths(dataset_name: str | None, output: str | None) -> DatasetPaths:
    name = dataset_name or str(getattr(project_config, "ACTIVE_DATASET"))
    datasets = getattr(project_config, "DATASETS")
    if name not in datasets:
        raise ValueError(f"Unknown dataset={name!r}; available={tuple(datasets)}")
    dataset_cfg = datasets[name]
    root = Path(dataset_cfg["ROOT"])
    output_checkpoint = Path(output) if output is not None else root / f"BEATs_{name}.pt"
    return DatasetPaths(
        name=name,
        root=root,
        train_csv=Path(dataset_cfg["TRAIN_CSV"]),
        eval_csv=Path(dataset_cfg["EVAL_CSV"]),
        test_csv=Path(dataset_cfg.get("TEST_CSV", root / "splits" / "test.csv")),
        full_csv=Path(dataset_cfg.get("FULL_CSV", root / "splits" / "full_dataset.csv")),
        label_mapping=Path(dataset_cfg["LABEL_MAPPING"]),
        output_checkpoint=output_checkpoint,
    )


def audio_config_from_project() -> AudioInputConfig:
    audio = getattr(project_config, "AUDIO", {})
    if not isinstance(audio, Mapping):
        audio = {}
    return AudioInputConfig(
        sample_rate=int(audio.get("TARGET_SR", 16000)),
        target_seconds=float(audio.get("TARGET_SECONDS", 10.0)),
        normalization=str(audio.get("NORMALIZATION", "peak")),  # type: ignore[arg-type]
        eps=float(audio.get("EPS", 1e-8)),
        deterministic_crop=str(audio.get("CROP_MODE", "center")),  # type: ignore[arg-type]
        fixed_crop_start=int(audio.get("FIXED_CROP_START", 0)),
    )


def load_label_mapping(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"label_mapping.json not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"label_mapping.json must be a non-empty mapping: {path}")
    mapping: dict[str, int] = {}
    for key, value in raw.items():
        label_id = int(value)
        if label_id < 0:
            raise ValueError(f"label_mapping contains negative id for {key!r}: {label_id}")
        mapping[str(key)] = label_id
    return mapping


def num_predictor_classes(label_mapping: Mapping[str, int]) -> int:
    return max(int(value) for value in label_mapping.values()) + 1


def validate_split_labels(split_csv: Path, *, num_classes: int, name: str) -> None:
    import pandas as pd

    frame = pd.read_csv(split_csv)
    if "label_id" not in frame.columns:
        raise KeyError(f"{name} split missing label_id column: {split_csv}")
    labels = pd.to_numeric(frame["label_id"], errors="raise").astype(int)
    invalid = labels[(labels < 0) | (labels >= int(num_classes))]
    if not invalid.empty:
        examples = invalid.head(8).tolist()
        raise ValueError(
            f"{name} split has label_id outside [0, {int(num_classes) - 1}], "
            f"examples={examples}"
        )

def _missing_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "<na>"}


def _optional_int(value: Any) -> int | None:
    if _missing_text(value):
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _has_text(value: Any) -> bool:
    return not _missing_text(value)


def _bucket_context(path: Path, row_number: int) -> str:
    return f"{path}:row={row_number}"


def _canonical_bucket_from_row(
    row: Mapping[str, Any],
    *,
    path: Path,
    row_number: int,
    allow_train_label_n: bool,
) -> str | None:
    bucket_value = row.get("bucket")
    shot_group_value = row.get("shot_group")
    train_label_n_value = row.get("train_label_n")
    if not _has_text(train_label_n_value):
        train_label_n_value = row.get("label_n")
    has_bucket = _has_text(bucket_value)
    has_shot_group = _has_text(shot_group_value)
    has_train_label_n = _has_text(train_label_n_value)

    if has_bucket or has_shot_group:
        try:
            result = read_canonical_bucket(
                bucket=[bucket_value] if has_bucket else None,
                shot_group=[shot_group_value] if has_shot_group else None,
                train_label_n=[train_label_n_value] if has_train_label_n else None,
                context=_bucket_context(path, row_number),
                allow_shot_group_fallback=False,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return result.bucket[0]

    if allow_train_label_n and has_train_label_n:
        try:
            return bucket_from_train_label_n(train_label_n_value)
        except ValueError as exc:
            raise ValueError(f"{_bucket_context(path, row_number)}: {exc}") from exc

    return None


def _label_buckets_from_csv(path: Path, *, allow_train_label_n: bool = True) -> dict[int, str]:
    if not path.exists():
        return {}
    buckets_by_label: dict[int, set[str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            label_id = _optional_int(row.get("label_id"))
            if label_id is None:
                continue
            bucket = _canonical_bucket_from_row(
                row,
                path=path,
                row_number=row_number,
                allow_train_label_n=allow_train_label_n,
            )
            if bucket is None:
                continue
            buckets_by_label.setdefault(label_id, set()).add(bucket)

    conflicts = {
        label_id: sorted(buckets)
        for label_id, buckets in buckets_by_label.items()
        if len(buckets) > 1
    }
    if conflicts:
        examples = "; ".join(
            f"label_id={label_id}: buckets={buckets}"
            for label_id, buckets in sorted(conflicts.items())[:8]
        )
        raise ValueError(f"{path}: inconsistent bucket metadata for the same label_id; {examples}")

    return {label_id: sorted(buckets)[0] for label_id, buckets in buckets_by_label.items()}

def _label_counts_from_csv(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not path.exists():
        return counts
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_id = _optional_int(row.get("label_id"))
            if label_id is not None:
                counts[label_id] += 1
    return counts


def infer_label_buckets(paths: DatasetPaths) -> dict[int, str]:
    result: dict[int, str] = {}
    for source in (paths.train_csv, paths.eval_csv, paths.full_csv):
        for label_id, bucket in _label_buckets_from_csv(source, allow_train_label_n=True).items():
            existing = result.get(int(label_id))
            if existing is not None and existing != bucket:
                raise ValueError(
                    f"{source}: bucket={bucket!r} for label_id={int(label_id)} conflicts with "
                    f"previous split metadata bucket={existing!r}"
                )
            result.setdefault(int(label_id), bucket)
    for label_id, count in _label_counts_from_csv(paths.train_csv).items():
        if int(label_id) in result:
            continue
        bucket = bucket_from_train_label_n(int(count))
        result.setdefault(int(label_id), bucket)
    return result


def label_names_by_id(label_mapping: Mapping[str, int]) -> dict[int, str]:
    return {int(label_id): str(label_name) for label_name, label_id in label_mapping.items()}

def epoch_log_path(dataset_name: str, log_dir: str | None) -> Path:
    root = Path(log_dir) if log_dir is not None else Path(project_config.BASE_DIR) / "outputs" / "log"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"train_beats_classifier_head_{dataset_name}_{stamp}.csv"

def per_class_log_path(epoch_path: Path) -> Path:
    return epoch_path.with_name(f"{epoch_path.stem}_per_class.csv")


def format_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


def make_grad_scaler(*, enabled: bool) -> Any | None:
    if not bool(enabled):
        return None
    amp_module = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if grad_scaler_cls is not None:
        try:
            return grad_scaler_cls("cuda", enabled=True)
        except TypeError:
            return grad_scaler_cls(enabled=True)
    cuda_amp = getattr(torch.cuda, "amp", None)
    grad_scaler_cls = getattr(cuda_amp, "GradScaler", None) if cuda_amp is not None else None
    if grad_scaler_cls is None:
        raise RuntimeError("CUDA AMP training requested, but torch GradScaler is unavailable")
    return grad_scaler_cls(enabled=True)


def resolve_run_git_commit(args: argparse.Namespace) -> str | None:
    commit = git_commit(required=not bool(args.allow_missing_git_commit))
    if commit is None:
        if not bool(args.skip_export_eval):
            raise RuntimeError(
                "--allow-missing-git-commit requires --skip-export-eval because export_eval writes formal "
                "manifests that still require a git commit. For a formal run, fix the checkout so "
                "`git rev-parse HEAD` works instead."
            )
        print(
            "[train-head] WARNING: git rev-parse HEAD failed; continuing only because "
            "--allow-missing-git-commit and --skip-export-eval were provided. "
            "This checkpoint is for debug only, not a formal Baseline artifact."
        )
    return commit


def elapsed_stage(name: str, started_at: float) -> float:
    now = time.perf_counter()
    print(f"[train-head] stage={name} elapsed={now - started_at:.2f}s", flush=True)
    return now


def begin_stage(name: str) -> float:
    print(f"[train-head] stage={name} started", flush=True)
    return time.perf_counter()


def resolve_state_output_path(paths: DatasetPaths, args: argparse.Namespace) -> Path | None:
    if bool(args.no_state_checkpoint):
        return None
    if args.state_output is not None:
        return Path(args.state_output)
    return paths.output_checkpoint.with_name(f"{paths.output_checkpoint.stem}.train_state.pt")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device=device)


def save_training_state(
    path: Path,
    *,
    model: BEATs,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    paths: DatasetPaths,
    args: argparse.Namespace,
    source_checkpoint_path: Path,
    git_commit_value: str | None,
    epoch: int,
    phase: str,
    next_batch_idx: int,
    best_selection_value: float,
    best_epoch: int,
    best_metrics: Mapping[str, Any],
    stale_epochs: int,
    train_metrics: Mapping[str, Any] | None = None,
    reason: str = "checkpoint",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact_type": "beats_classifier_head_training_state",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": str(reason),
        "dataset_name": paths.name,
        "source_checkpoint": str(source_checkpoint_path.resolve(strict=False)),
        "output_checkpoint": str(paths.output_checkpoint.resolve(strict=False)),
        "git_commit": git_commit_value,
        "seed": int(args.seed),
        "selection_metric": str(args.selection_metric),
        "epoch": int(epoch),
        "phase": str(phase),
        "next_batch_idx": int(next_batch_idx),
        "best_selection_value": float(best_selection_value),
        "best_epoch": int(best_epoch),
        "best_metrics": dict(best_metrics),
        "stale_epochs": int(stale_epochs),
        "train_metrics": dict(train_metrics) if train_metrics is not None else None,
        "predictor": model.predictor.state_dict(),
        "predictor_dropout": model.predictor_dropout.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    print(
        "[train-head] saved training state "
        f"path={path} phase={phase} epoch={int(epoch)} next_batch_idx={int(next_batch_idx)} reason={reason}"
    )


def load_training_state(
    path: Path,
    *,
    model: BEATs,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    paths: DatasetPaths,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"resume state not found: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError(f"resume state must be a mapping, got {type(state)!r}")
    if state.get("artifact_type") != "beats_classifier_head_training_state":
        raise ValueError(f"unsupported resume state artifact_type={state.get('artifact_type')!r}")
    if str(state.get("dataset_name")) != paths.name:
        raise ValueError(f"resume state dataset={state.get('dataset_name')!r} does not match current dataset={paths.name!r}")
    if int(state.get("seed", -1)) != int(args.seed):
        raise ValueError(f"resume state seed={state.get('seed')!r} does not match current seed={args.seed!r}")
    if str(state.get("selection_metric")) != str(args.selection_metric):
        raise ValueError(
            f"resume state selection_metric={state.get('selection_metric')!r} "
            f"does not match current selection_metric={args.selection_metric!r}"
        )
    model.predictor.load_state_dict(state["predictor"])
    model.predictor_dropout.load_state_dict(state.get("predictor_dropout", {}), strict=False)
    optimizer.load_state_dict(state["optimizer"])
    move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return dict(state)


def initial_selection_metric_value(metric: str) -> float:
    return float("inf") if metric == "eval_loss" else float("-inf")


def selection_metric_value(
    metric: str,
    *,
    eval_metrics: Mapping[str, Any],
    eval_breakdown: Mapping[str, float],
) -> float:
    if metric == "eval_loss":
        return float(eval_metrics["loss"])
    if metric == "eval_acc":
        return float(eval_metrics["accuracy"])
    if metric == "eval_macro_f1_label_universe_nanmean":
        return float(eval_breakdown.get("eval_macro_f1_label_universe_nanmean", float("nan")))
    raise ValueError(f"unsupported selection_metric={metric!r}")



def selection_metric_improved(metric: str, current: float, best: float) -> bool:
    if not math.isfinite(current):
        return False
    if not math.isfinite(best):
        return True
    if metric == "eval_loss":
        return current < best
    if metric in {"eval_acc", "eval_macro_f1_label_universe_nanmean"}:
        return current > best
    raise ValueError(f"unsupported selection_metric={metric!r}")

def build_loader(
    paths: DatasetPaths,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle_seed: int | None = None,
) -> DataLoader:
    if split == "train":
        split_csv = paths.train_csv
        mode = "train"
        shuffle = True
    elif split == "eval":
        split_csv = paths.eval_csv
        mode = "eval"
        shuffle = False
    elif split == "test":
        split_csv = paths.test_csv
        mode = "test"
        shuffle = False
    else:
        raise ValueError(f"unsupported split={split!r}")

    dataset = AudioInputDataset(
        split_csv,
        mode=mode,
        dataset_root=paths.root,
        project_root=PROJECT_ROOT,
        audio_config=audio_config_from_project(),
        global_seed=int(seed),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    generator = None
    if shuffle and shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
        generator=generator,
    )


def load_source_model(checkpoint_path: Path, *, num_classes: int, device: torch.device) -> tuple[BEATs, dict[str, Any], dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"source BEATs checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint)!r}")
    if "cfg" not in checkpoint or "model" not in checkpoint:
        raise KeyError("checkpoint must contain 'cfg' and 'model'")

    original_cfg = copy.deepcopy(dict(checkpoint["cfg"]))
    model = BEATs(BEATsConfig(original_cfg))
    if not hasattr(model, "predictor_dropout"):
        model.predictor_dropout = nn.Dropout(float(original_cfg.get("predictor_dropout", 0.1)))
    model.predictor = nn.Linear(int(model.cfg.encoder_embed_dim), int(num_classes))
    model.cfg.finetuned_model = True
    model.cfg.predictor_class = int(num_classes)

    source_state = checkpoint["model"]
    if not isinstance(source_state, Mapping):
        raise TypeError(f"checkpoint['model'] must be a state_dict mapping, got {type(source_state)!r}")
    backbone_state = {
        str(name): tensor
        for name, tensor in source_state.items()
        if not str(name).startswith("predictor.")
    }
    load_result = model.load_state_dict(backbone_state, strict=False)
    unexpected = [name for name in load_result.unexpected_keys if not str(name).startswith("predictor.")]
    missing = [name for name in load_result.missing_keys if not str(name).startswith("predictor.")]
    if missing or unexpected:
        raise RuntimeError(
            "source checkpoint backbone does not match BEATs model: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.to(device)
    return model, dict(checkpoint), original_cfg


def freeze_backbone_train_predictor(model: BEATs) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.predictor.parameters():
        parameter.requires_grad = True


def backbone_features(model: BEATs, waveform: Tensor, padding_mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
    with torch.no_grad():
        fbank = model.preprocess(waveform)
        if padding_mask is not None:
            padding_mask = model.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = model.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = model.layer_norm(features)

        if padding_mask is not None:
            padding_mask = model.forward_padding_mask(features, padding_mask)

        if model.post_extract_proj is not None:
            features = model.post_extract_proj(features)

        x = model.dropout_input(features)
        x, _layer_results = model.encoder(x, padding_mask=padding_mask)
    return x.detach(), padding_mask


def forward_logits(
    model: BEATs,
    waveform: Tensor,
    padding_mask: Tensor | None,
    *,
    predictor_amp: bool = False,
) -> Tensor:
    x, feature_padding_mask = backbone_features(model, waveform, padding_mask)

    def predictor_autocast_context() -> Any:
        return torch.autocast(device_type="cuda") if predictor_amp and x.device.type == "cuda" else nullcontext()

    with predictor_autocast_context():
        x = model.predictor_dropout(x)
        logits = model.predictor(x)
    if feature_padding_mask is not None and feature_padding_mask.any():
        valid = ~feature_padding_mask
        logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
        denom = valid.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype=logits.dtype)
        return (logits.sum(dim=1) / denom).float()
    return logits.mean(dim=1).float()


def assert_finite_tensor(
    name: str,
    tensor: Tensor,
    *,
    phase: str,
    epoch: int,
    batch_idx: int,
    sample_ids: Sequence[Any] | None = None,
) -> None:
    values = tensor.detach()
    if bool(torch.isfinite(values).all().item()):
        return
    finite_mask = torch.isfinite(values)
    nonfinite_count = int((~finite_mask).sum().item())
    if sample_ids is None:
        preview: list[Any] = []
    elif isinstance(sample_ids, Tensor):
        preview = list(sample_ids.detach().cpu().tolist())[:8]
    elif isinstance(sample_ids, (str, bytes)):
        preview = [sample_ids]
    else:
        preview = list(sample_ids)[:8]
    raise FloatingPointError(
        f"{phase} epoch={epoch} batch={batch_idx} produced non-finite {name} "
        f"(nonfinite_count={nonfinite_count}, shape={tuple(values.shape)}, sample_id_preview={preview}). "
        "For this head-training script, first retry without --amp; if it still fails, lower --lr and inspect the listed samples."
    )


def set_head_training_mode(model: BEATs) -> None:
    model.eval()
    model.predictor.train()
    model.predictor_dropout.train()


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value.to(device=device, non_blocking=True)
        else:
            result[key] = value
    return result


def run_epoch(
    model: BEATs,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    scaler: Any | None = None,
    phase: str,
    epoch: int,
    max_batches: int | None = None,
    skip_batches: int = 0,
    on_batch_end: Callable[[int, int], None] | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    if training:
        set_head_training_mode(model)
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    observed_labels: list[int] = []
    observed_predictions: list[int] = []
    num_batches = 0
    skipped_batches = max(0, int(skip_batches))
    autocast_enabled = bool(amp and device.type == "cuda")
    active_scaler = scaler if training and autocast_enabled else None
    if training and autocast_enabled and active_scaler is None:
        active_scaler = make_grad_scaler(enabled=True)

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx <= skipped_batches:
            continue
        batch = move_batch_to_device(batch, device)
        labels = batch["label_id"].long()
        sample_ids = batch.get("sample_id")
        if training:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(
                model,
                batch["waveform"],
                batch.get("padding_mask"),
                predictor_amp=autocast_enabled,
            )
            assert_finite_tensor("logits", logits, phase=phase, epoch=epoch, batch_idx=batch_idx, sample_ids=sample_ids)
            loss = criterion(logits, labels)
            assert_finite_tensor("loss", loss, phase=phase, epoch=epoch, batch_idx=batch_idx, sample_ids=sample_ids)
            if active_scaler is not None:
                active_scaler.scale(loss).backward()
                active_scaler.step(optimizer)
                active_scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.inference_mode():
                logits = forward_logits(
                    model,
                    batch["waveform"],
                    batch.get("padding_mask"),
                    predictor_amp=autocast_enabled,
                )
                assert_finite_tensor("logits", logits, phase=phase, epoch=epoch, batch_idx=batch_idx, sample_ids=sample_ids)
                loss = criterion(logits, labels)
                assert_finite_tensor("loss", loss, phase=phase, epoch=epoch, batch_idx=batch_idx, sample_ids=sample_ids)

        predictions = logits.detach().argmax(dim=1)
        batch_size = int(labels.numel())
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        num_batches += 1
        if collect_predictions:
            observed_labels.extend(int(value) for value in labels.detach().cpu().tolist())
            observed_predictions.extend(int(value) for value in predictions.cpu().tolist())
        if on_batch_end is not None:
            on_batch_end(batch_idx, num_batches)
        if max_batches is not None and num_batches >= int(max_batches):
            break

    if total_samples <= 0:
        raise RuntimeError(f"loader produced no samples after skip_batches={skipped_batches}")
    metrics: dict[str, Any] = {
        "loss": total_loss / float(total_samples),
        "accuracy": total_correct / float(total_samples),
        "num_samples": float(total_samples),
        "num_batches": float(num_batches),
        "skipped_batches": float(skipped_batches),
    }
    if collect_predictions:
        metrics["labels"] = observed_labels
        metrics["predictions"] = observed_predictions
    return metrics


def _mean_or_nan(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / float(len(finite)) if finite else float("nan")


def _class_f1(
    label_id: int,
    *,
    true_count: Mapping[int, int],
    pred_count: Mapping[int, int],
    correct_count: Mapping[int, int],
    absent_value: float = 0.0,
) -> float:
    support = int(true_count.get(int(label_id), 0))
    if support <= 0:
        return float(absent_value)
    correct = int(correct_count.get(int(label_id), 0))
    false_positive = max(0, int(pred_count.get(int(label_id), 0)) - correct)
    false_negative = max(0, support - correct)
    denominator = (2 * correct) + false_positive + false_negative
    return (2 * correct) / float(denominator) if denominator > 0 else float(absent_value)

def compute_eval_breakdown(
    eval_metrics: Mapping[str, Any],
    *,
    label_names: Mapping[int, str],
    label_buckets: Mapping[int, str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    labels = [int(value) for value in eval_metrics.get("labels", [])]
    predictions = [int(value) for value in eval_metrics.get("predictions", [])]
    if not labels or len(labels) != len(predictions):
        raise RuntimeError("eval predictions were not collected correctly")

    true_count: Counter[int] = Counter(labels)
    pred_count: Counter[int] = Counter(predictions)
    correct_count: Counter[int] = Counter()
    bucket_count: Counter[str] = Counter()
    bucket_correct: Counter[str] = Counter()

    for label_id, prediction in zip(labels, predictions):
        label_id = int(label_id)
        prediction = int(prediction)
        is_correct = prediction == label_id
        if is_correct:
            correct_count[label_id] += 1
        bucket = label_buckets.get(label_id, "unknown")
        if bucket in BASE_BUCKETS:
            bucket_count[bucket] += 1
            if is_correct:
                bucket_correct[bucket] += 1

    per_class_rows: list[dict[str, Any]] = []
    accuracy_values: list[float] = []
    f1_values_label_universe: list[float] = []
    f1_values_observed: list[float] = []
    f1_values_legacy_zero_absent: list[float] = []
    all_label_ids = sorted(int(label_id) for label_id in label_names.keys())
    for label_id in all_label_ids:
        count = int(true_count[label_id])
        correct = int(correct_count[label_id])
        false_positive = max(0, int(pred_count[label_id]) - correct)
        false_negative = max(0, count - correct)
        accuracy = correct / float(count) if count > 0 else float("nan")
        precision = correct / float(correct + false_positive) if (correct + false_positive) > 0 else float("nan")
        recall = correct / float(correct + false_negative) if (correct + false_negative) > 0 else float("nan")
        f1 = _class_f1(
            label_id,
            true_count=true_count,
            pred_count=pred_count,
            correct_count=correct_count,
            absent_value=float("nan"),
        )
        f1_legacy = _class_f1(
            label_id,
            true_count=true_count,
            pred_count=pred_count,
            correct_count=correct_count,
            absent_value=0.0,
        )
        if count > 0:
            accuracy_values.append(float(accuracy))
            f1_values_observed.append(float(f1))
        f1_values_label_universe.append(float(f1))
        f1_values_legacy_zero_absent.append(float(f1_legacy))
        per_class_rows.append(
            {
                "label_id": label_id,
                "label_name": label_names.get(label_id, str(label_id)),
                "bucket": label_buckets.get(label_id, "unknown"),
                "count": count,
                "correct": correct,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f1_legacy_zero_absent_diagnostic": f1_legacy,
            }
        )
    breakdown: dict[str, float] = {
        "eval_macro_acc": _mean_or_nan(accuracy_values),
        "eval_macro_f1_label_universe_nanmean": _mean_or_nan(f1_values_label_universe),
        "eval_macro_f1_observed_classes": _mean_or_nan(f1_values_observed),
        "eval_macro_f1_legacy_zero_absent_diagnostic": _mean_or_nan(f1_values_legacy_zero_absent),
        "eval_num_classes_total": float(len(all_label_ids)),
        "eval_num_classes_present": float(len(true_count)),
    }
    for bucket in BASE_BUCKETS:
        count = int(bucket_count[bucket])
        correct = int(bucket_correct[bucket])
        breakdown[f"eval_{bucket}_count"] = float(count)
        breakdown[f"eval_{bucket}_acc"] = correct / float(count) if count > 0 else float("nan")

    tail_bucket_mask = analysis_group_mask(BASE_BUCKETS, "tail")
    tail_count = sum(int(bucket_count[bucket]) for bucket, include in zip(BASE_BUCKETS, tail_bucket_mask) if include)
    tail_correct = sum(int(bucket_correct[bucket]) for bucket, include in zip(BASE_BUCKETS, tail_bucket_mask) if include)
    breakdown["eval_tail_count"] = float(tail_count)
    breakdown["eval_tail_acc"] = tail_correct / float(tail_count) if tail_count > 0 else float("nan")
    return breakdown, per_class_rows


def checkpoint_payload(
    *,
    source_checkpoint: Mapping[str, Any],
    source_checkpoint_path: Path,
    original_cfg: Mapping[str, Any],
    model: BEATs,
    label_mapping: Mapping[str, int],
    paths: DatasetPaths,
    args: argparse.Namespace,
    epoch: int,
    eval_metrics: Mapping[str, float],
    eval_breakdown: Mapping[str, float],
    git_commit_value: str | None,
) -> dict[str, Any]:
    save_cfg = copy.deepcopy(dict(original_cfg))
    save_cfg["finetuned_model"] = True
    save_cfg["predictor_class"] = int(model.cfg.predictor_class)
    label_mapping_hash = file_sha256(paths.label_mapping)
    source_checkpoint_hash = file_sha256(source_checkpoint_path)
    checkpoint_identity_payload = {
        "model_kind": "fp32_teacher",
        "cfg_hash": canonical_json_hash(save_cfg),
        "label_mapping_hash": label_mapping_hash,
        "source_beats_checkpoint": str(source_checkpoint_path.resolve(strict=False)),
        "source_beats_checkpoint_hash": source_checkpoint_hash,
        "pooling": "mask_aware_mean",
        "seed": int(args.seed),
        "git_commit": git_commit_value,
    }
    checkpoint_run_identity_hash = canonical_json_hash(checkpoint_identity_payload)
    payload = {
        "cfg": save_cfg,
        "model": model.state_dict(),
        "label_mapping": dict(label_mapping),
        "label_mapping_hash": label_mapping_hash,
        "dataset_name": paths.name,
        "artifact_type": "baseline_fp32_teacher_checkpoint",
        "model_kind": "fp32_teacher",
        "source_beats_checkpoint_hash": source_checkpoint_hash,
        "source_checkpoint_sha256": source_checkpoint_hash,
        "pooling": "mask_aware_mean",
        "seed": int(args.seed),
        "git_commit": checkpoint_identity_payload["git_commit"],
        "run_identity_hash": checkpoint_run_identity_hash,
        "run_identity_payload": checkpoint_identity_payload,
        "checkpoint_role": "dataset_fp32_teacher",
        "dataset_fp32_teacher_checkpoint": str(paths.output_checkpoint.resolve(strict=False)),
        "source_checkpoint": str(source_checkpoint_path.resolve(strict=False)),
        "source_checkpoint_role": "beats_pretrained_backbone",
        "baseline_pipeline": {
            "stageable_as_fp32_teacher": True,
            "staging_note": "Baseline runners copy or verify this dataset checkpoint into the per-seed fp32_teacher slot before INT4/KD stages.",
        },
        "classifier_head_training": {
            "frozen_backbone": True,
            "epoch": int(epoch),
            "best_eval_loss": float(eval_metrics["loss"]),
            "best_eval_accuracy": float(eval_metrics["accuracy"]),
            "best_eval_macro_f1_label_universe_nanmean": float(eval_breakdown.get("eval_macro_f1_label_universe_nanmean", float("nan"))),
            "best_eval_macro_f1_scope": "label_universe_support0_nan_nanmean",
            "best_eval_macro_f1_observed_classes": float(eval_breakdown.get("eval_macro_f1_observed_classes", float("nan"))),
            "best_eval_macro_f1_legacy_zero_absent_diagnostic": float(eval_breakdown.get("eval_macro_f1_legacy_zero_absent_diagnostic", float("nan"))),
            "eval_num_classes_present": int(eval_breakdown.get("eval_num_classes_present", 0.0)),
            "eval_num_classes_total": int(eval_breakdown.get("eval_num_classes_total", 0.0)),
            "train_csv": str(paths.train_csv.resolve(strict=False)),
            "eval_csv": str(paths.eval_csv.resolve(strict=False)),
            "label_mapping": str(paths.label_mapping.resolve(strict=False)),
            "train_audio_mode": "train",
            "eval_audio_mode": "eval",
            "selection_metric": str(args.selection_metric),
            "best_selection_metric_value": float(selection_metric_value(
                str(args.selection_metric),
                eval_metrics=eval_metrics,
                eval_breakdown=eval_breakdown,
            )),
            "epochs_requested": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size or args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(args.label_smoothing),
            "seed": int(args.seed),
            "amp_requested": bool(args.amp),
            "cuda_amp_training_requires_grad_scaler": True,
        },
    }
    if "cfg" in source_checkpoint and "model" in source_checkpoint:
        payload["source_checkpoint_keys"] = sorted(str(key) for key in source_checkpoint.keys())
    return payload


def append_epoch_log(
    path: Path,
    *,
    epoch: int,
    train_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any],
    eval_breakdown: Mapping[str, float],
) -> None:
    fieldnames = (
        "epoch",
        "train_loss",
        "train_acc",
        "eval_loss",
        "eval_acc",
        "eval_many_acc",
        "eval_medium_acc",
        "eval_few_acc",
        "eval_tail_acc",
        "eval_many_count",
        "eval_medium_count",
        "eval_few_count",
        "eval_tail_count",
        "eval_macro_acc",
        "eval_macro_f1_label_universe_nanmean",
        "eval_macro_f1_observed_classes",
        "eval_macro_f1_legacy_zero_absent_diagnostic",
        "eval_num_classes_present",
        "eval_num_classes_total",
    )
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": int(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_acc": float(train_metrics["accuracy"]),
                "eval_loss": float(eval_metrics["loss"]),
                "eval_acc": float(eval_metrics["accuracy"]),
                "eval_many_acc": float(eval_breakdown.get("eval_many_acc", float("nan"))),
                "eval_medium_acc": float(eval_breakdown.get("eval_medium_acc", float("nan"))),
                "eval_few_acc": float(eval_breakdown.get("eval_few_acc", float("nan"))),
                "eval_tail_acc": float(eval_breakdown.get("eval_tail_acc", float("nan"))),
                "eval_many_count": int(eval_breakdown.get("eval_many_count", 0.0)),
                "eval_medium_count": int(eval_breakdown.get("eval_medium_count", 0.0)),
                "eval_few_count": int(eval_breakdown.get("eval_few_count", 0.0)),
                "eval_tail_count": int(eval_breakdown.get("eval_tail_count", 0.0)),
                "eval_macro_acc": float(eval_breakdown.get("eval_macro_acc", float("nan"))),
                "eval_macro_f1_label_universe_nanmean": float(eval_breakdown.get("eval_macro_f1_label_universe_nanmean", float("nan"))),
                "eval_macro_f1_observed_classes": float(eval_breakdown.get("eval_macro_f1_observed_classes", float("nan"))),
                "eval_macro_f1_legacy_zero_absent_diagnostic": float(eval_breakdown.get("eval_macro_f1_legacy_zero_absent_diagnostic", float("nan"))),
                "eval_num_classes_present": int(eval_breakdown.get("eval_num_classes_present", 0.0)),
                "eval_num_classes_total": int(eval_breakdown.get("eval_num_classes_total", 0.0)),
            }
        )


def append_per_class_log(path: Path, *, epoch: int, per_class_rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "epoch",
        "label_id",
        "label_name",
        "bucket",
        "count",
        "correct",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_legacy_zero_absent_diagnostic",
    )
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in per_class_rows:
            writer.writerow(
                {
                    "epoch": int(epoch),
                    "label_id": int(row["label_id"]),
                    "label_name": str(row["label_name"]),
                    "bucket": str(row["bucket"]),
                    "count": int(row["count"]),
                    "correct": int(row["correct"]),
                    "accuracy": float(row["accuracy"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "f1_legacy_zero_absent_diagnostic": float(row.get("f1_legacy_zero_absent_diagnostic", float("nan"))),
                }
            )


def main() -> dict[str, Any]:
    args = parse_args()
    stage_started_at = begin_stage("preflight")
    seed_everything(int(args.seed))

    paths = dataset_paths(args.dataset, args.output)
    if paths.output_checkpoint.exists() and not args.overwrite and args.resume_state is None:
        raise FileExistsError(f"output checkpoint already exists; pass --overwrite to replace it: {paths.output_checkpoint}")
    paths.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state_output_path = resolve_state_output_path(paths, args)
    run_git_commit = resolve_run_git_commit(args)
    stage_started_at = elapsed_stage("preflight", stage_started_at)
    stage_started_at = begin_stage("label_validation")

    label_mapping = load_label_mapping(paths.label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.train_csv, num_classes=num_classes, name="train")
    validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    if not bool(args.skip_export_eval):
        validate_split_labels(paths.test_csv, num_classes=num_classes, name="test")
    label_names = label_names_by_id(label_mapping)
    label_buckets = infer_label_buckets(paths)
    stage_started_at = elapsed_stage("label_validation", stage_started_at)
    stage_started_at = begin_stage("model_load")

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device={device}, but CUDA is not available")

    checkpoint_path = resolve_source_checkpoint_path(args.checkpoint)
    model, source_checkpoint, original_cfg = load_source_model(checkpoint_path, num_classes=num_classes, device=device)
    freeze_backbone_train_predictor(model)
    stage_started_at = elapsed_stage("model_load", stage_started_at)
    stage_started_at = begin_stage("eval_loader_build")

    eval_loader = build_loader(
        paths,
        split="eval",
        batch_size=int(args.eval_batch_size or args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    stage_started_at = elapsed_stage("eval_loader_build", stage_started_at)
    stage_started_at = begin_stage("optimizer_and_state")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    amp_training_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = make_grad_scaler(enabled=amp_training_enabled)

    best_selection_value = initial_selection_metric_value(args.selection_metric)
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    stale_epochs = 0
    log_path = epoch_log_path(paths.name, args.log_dir)
    per_class_path = per_class_log_path(log_path)
    start_epoch = 1
    resume_phase = "train"
    resume_skip_batches = 0
    pending_train_metrics: dict[str, Any] | None = None
    if args.resume_state is not None:
        resume_state = load_training_state(
            Path(args.resume_state),
            model=model,
            optimizer=optimizer,
            scaler=grad_scaler,
            paths=paths,
            args=args,
            device=device,
        )
        best_selection_value = float(resume_state.get("best_selection_value", best_selection_value))
        best_epoch = int(resume_state.get("best_epoch", best_epoch))
        best_metrics = dict(resume_state.get("best_metrics") or {})
        stale_epochs = int(resume_state.get("stale_epochs", stale_epochs))
        start_epoch = int(resume_state.get("epoch", 1))
        resume_phase = str(resume_state.get("phase", "train"))
        if resume_phase not in {"train", "eval"}:
            raise ValueError(f"unsupported resume phase={resume_phase!r}")
        if resume_phase == "train":
            resume_skip_batches = max(0, int(resume_state.get("next_batch_idx", 1)) - 1)
        else:
            train_metrics_state = resume_state.get("train_metrics")
            if not isinstance(train_metrics_state, Mapping):
                raise ValueError("resume state phase='eval' is missing train_metrics")
            pending_train_metrics = dict(train_metrics_state)
        print(
            "[train-head] resumed state "
            f"path={args.resume_state} phase={resume_phase} epoch={start_epoch} "
            f"skip_batches={resume_skip_batches} best_epoch={best_epoch}"
        )
    elif state_output_path is not None:
        save_training_state(
            state_output_path,
            model=model,
            optimizer=optimizer,
            scaler=grad_scaler,
            paths=paths,
            args=args,
            source_checkpoint_path=checkpoint_path,
            git_commit_value=run_git_commit,
            epoch=1,
            phase="train",
            next_batch_idx=1,
            best_selection_value=best_selection_value,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            stale_epochs=stale_epochs,
            reason="initialized",
        )
    stage_started_at = elapsed_stage("optimizer_and_state", stage_started_at)

    print(f"[train-head] dataset={paths.name} num_classes={num_classes} source={checkpoint_path} output={paths.output_checkpoint}")
    print(f"[train-head] epoch log: {log_path}")
    print(f"[train-head] per-class log: {per_class_path}")
    if state_output_path is not None:
        print(f"[train-head] training state: {state_output_path} save_every_batches={int(args.save_state_every_batches)}")
    print(f"[train-head] frozen backbone, trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"[train-head] selection_metric={args.selection_metric}")
    if bool(args.amp):
        print(f"[train-head] amp_requested=True cuda_amp_training={amp_training_enabled} grad_scaler_enabled={grad_scaler is not None}")
    if args.max_train_batches is not None or args.max_eval_batches is not None:
        print(
            "[train-head] debug batch limits "
            f"max_train_batches={args.max_train_batches} "
            f"max_eval_batches={args.max_eval_batches}"
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        epoch_stage_started_at = begin_stage(f"epoch_{epoch:03d}_train")
        if epoch == start_epoch and resume_phase == "eval":
            if pending_train_metrics is None:
                raise RuntimeError("resume requested eval phase but no pending train metrics are available")
            train_metrics = dict(pending_train_metrics)
            print(f"[train-head] resume epoch={epoch:03d}: train already complete, running eval next")
        else:
            train_loader = build_loader(
                paths,
                split="train",
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed),
                shuffle_seed=int(args.seed) + (int(epoch) * 1_000_003),
            )
            train_dataset = getattr(train_loader, "dataset", None)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            current_skip_batches = resume_skip_batches if epoch == start_epoch and resume_phase == "train" else 0
            if current_skip_batches > 0:
                print(f"[train-head] resume epoch={epoch:03d}: skipping {current_skip_batches} completed train batches")

            def save_batch_state(batch_idx: int, processed_batches: int) -> None:
                del processed_batches
                interval = int(args.save_state_every_batches)
                if state_output_path is None or interval <= 0 or batch_idx % interval != 0:
                    return
                save_training_state(
                    state_output_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=grad_scaler,
                    paths=paths,
                    args=args,
                    source_checkpoint_path=checkpoint_path,
                    git_commit_value=run_git_commit,
                    epoch=epoch,
                    phase="train",
                    next_batch_idx=batch_idx + 1,
                    best_selection_value=best_selection_value,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    stale_epochs=stale_epochs,
                    reason=f"train_batch_{batch_idx}",
                )

            train_metrics = run_epoch(
                model,
                train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                amp=bool(args.amp),
                scaler=grad_scaler,
                phase="train",
                epoch=epoch,
                max_batches=args.max_train_batches,
                skip_batches=current_skip_batches,
                on_batch_end=save_batch_state,
            )
            if state_output_path is not None:
                save_training_state(
                    state_output_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=grad_scaler,
                    paths=paths,
                    args=args,
                    source_checkpoint_path=checkpoint_path,
                    git_commit_value=run_git_commit,
                    epoch=epoch,
                    phase="eval",
                    next_batch_idx=1,
                    best_selection_value=best_selection_value,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    stale_epochs=stale_epochs,
                    train_metrics=train_metrics,
                    reason="train_epoch_complete",
                )
        elapsed_stage(f"epoch_{epoch:03d}_train", epoch_stage_started_at)
        epoch_stage_started_at = begin_stage(f"epoch_{epoch:03d}_eval")
        eval_metrics = run_epoch(
            model,
            eval_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            amp=bool(args.amp),
            scaler=None,
            phase="eval",
            epoch=epoch,
            max_batches=args.max_eval_batches,
            collect_predictions=True,
        )
        elapsed_stage(f"epoch_{epoch:03d}_eval", epoch_stage_started_at)
        resume_phase = "train"
        resume_skip_batches = 0
        pending_train_metrics = None
        eval_breakdown, per_class_rows = compute_eval_breakdown(
            eval_metrics,
            label_names=label_names,
            label_buckets=label_buckets,
        )
        selected_value = selection_metric_value(
            args.selection_metric,
            eval_metrics=eval_metrics,
            eval_breakdown=eval_breakdown,
        )
        if not math.isfinite(selected_value):
            raise RuntimeError(f"selection metric {args.selection_metric!r} produced non-finite value")
        print(
            "[train-head] "
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_loss={eval_metrics['loss']:.6f} "
            f"eval_acc={eval_metrics['accuracy']:.4f} "
            f"{args.selection_metric}={selected_value:.6f}"
        )
        print(
            "[train-head] eval_breakdown "
            f"many_acc={format_metric(eval_breakdown.get('eval_many_acc'))} "
            f"many_n={int(eval_breakdown.get('eval_many_count', 0.0))} "
            f"medium_acc={format_metric(eval_breakdown.get('eval_medium_acc'))} "
            f"medium_n={int(eval_breakdown.get('eval_medium_count', 0.0))} "
            f"few_acc={format_metric(eval_breakdown.get('eval_few_acc'))} "
            f"few_n={int(eval_breakdown.get('eval_few_count', 0.0))} "
            f"tail_acc={format_metric(eval_breakdown.get('eval_tail_acc'))} "
            f"tail_n={int(eval_breakdown.get('eval_tail_count', 0.0))} "
            f"macro_acc={format_metric(eval_breakdown.get('eval_macro_acc'))} "
            f"macro_f1_label_universe_nanmean={format_metric(eval_breakdown.get('eval_macro_f1_label_universe_nanmean'))}"
        )
        append_epoch_log(
            log_path,
            epoch=epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            eval_breakdown=eval_breakdown,
        )
        append_per_class_log(per_class_path, epoch=epoch, per_class_rows=per_class_rows)

        if selection_metric_improved(args.selection_metric, selected_value, best_selection_value):
            best_selection_value = float(selected_value)
            best_epoch = int(epoch)
            best_metrics = dict(eval_metrics)
            best_metrics.update(eval_breakdown)
            stale_epochs = 0
            torch.save(
                checkpoint_payload(
                    source_checkpoint=source_checkpoint,
                    source_checkpoint_path=checkpoint_path,
                    original_cfg=original_cfg,
                    model=model,
                    label_mapping=label_mapping,
                    paths=paths,
                    args=args,
                    epoch=epoch,
                    eval_metrics=eval_metrics,
                    eval_breakdown=eval_breakdown,
                    git_commit_value=run_git_commit,
                ),
                paths.output_checkpoint,
            )
            print(f"[train-head] saved best checkpoint: {paths.output_checkpoint}")
        else:
            stale_epochs += 1

        if state_output_path is not None:
            save_training_state(
                state_output_path,
                model=model,
                optimizer=optimizer,
                scaler=grad_scaler,
                paths=paths,
                args=args,
                source_checkpoint_path=checkpoint_path,
                git_commit_value=run_git_commit,
                epoch=epoch + 1,
                phase="train",
                next_batch_idx=1,
                best_selection_value=best_selection_value,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                stale_epochs=stale_epochs,
                reason="epoch_complete",
            )

        if int(args.patience) > 0 and stale_epochs >= int(args.patience):
            print(f"[train-head] early stop after {stale_epochs} stale epochs")
            break

    export_summary: dict[str, Any] | None = None
    if not bool(args.skip_export_eval):
        if best_epoch < 0:
            raise RuntimeError("training did not produce a best checkpoint; cannot run export_eval")
        export_summary = run_export_eval(
            paths=paths,
            args=args,
            label_buckets=label_buckets,
            device=device,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
        )

    return {
        "dataset": paths.name,
        "output_checkpoint": paths.output_checkpoint,
        "state_output": state_output_path,
        "log_path": log_path,
        "per_class_log_path": per_class_path,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_eval_loss": best_metrics.get("loss", float("nan")),
        "best_eval_accuracy": best_metrics.get("accuracy", float("nan")),
        "best_eval_macro_f1_label_universe_nanmean": best_metrics.get("eval_macro_f1_label_universe_nanmean", float("nan")),
        "selection_metric": args.selection_metric,
        "best_selection_metric_value": best_selection_value,
        "export_eval": export_summary,
    }


if __name__ == "__main__":
    summary = main()
    print(
        "[train-head] done "
        f"best_epoch={summary['best_epoch']} "
        f"best_eval_loss={summary['best_eval_loss']:.6f} "
        f"best_eval_acc={summary['best_eval_accuracy']:.4f} "
        f"selection_metric={summary['selection_metric']} "
        f"best_selection_value={summary['best_selection_metric_value']:.6f} "
        f"output={summary['output_checkpoint']}"
    )
