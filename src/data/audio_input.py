from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
from torch.utils.data import Dataset
"""
负责实现统一音频输入规则
读取原始音频
转 mono
重采样到 16000
peak normalize 到 [-1, 1]
按 target_seconds 生成固定长度 waveform
不足长度右侧 padding，padding 值必须固定为 0
同步生成 padding_mask
返回 mono waveform，shape = [target_samples]；DataLoader batch 后 shape = [B, target_samples]；
它还要区分两种 crop 模式：
train：random crop
eval / test / export / geometry / Quant Injection：center crop 或固定 crop
"""
_REQUIRED_AUDIO_CONFIG_KEYS = (
    "TARGET_SR",
    "TARGET_SECONDS",
    "MONO",
    "NORMALIZATION",
    "EPS",
    "CROP_MODE",
    "TARGET_SAMPLES_RULE",
    "PADDING_VALUE",
    "PADDING_MASK_TRUE_MEANS",
    "FIXED_CROP_START",
    "RANDOM_CROP_GLOBAL_SEED",
    "INCLUDE_AUDIO_STATS_DEFAULT",
    "RESAMPLER_CACHE_MAX_SIZE",
    "DETERMINISTIC_CROP_START_CACHE_MAX_SIZE",
)


def _load_project_audio_defaults() -> dict[str, Any]:
    try:
        import config as project_config
    except ModuleNotFoundError as exc:
        if exc.name != "config":
            raise
        raise RuntimeError("audio_input.py requires project config.py with AUDIO protocol") from exc

    audio_cfg = getattr(project_config, "AUDIO", {})
    if not isinstance(audio_cfg, Mapping):
        raise RuntimeError("config.AUDIO must be a mapping")

    missing = [key for key in _REQUIRED_AUDIO_CONFIG_KEYS if key not in audio_cfg]
    if missing:
        raise RuntimeError(f"config.AUDIO missing required keys: {missing}")

    return {key: audio_cfg[key] for key in _REQUIRED_AUDIO_CONFIG_KEYS}


_AUDIO_DEFAULTS = _load_project_audio_defaults()
DEFAULT_SAMPLE_RATE = int(_AUDIO_DEFAULTS["TARGET_SR"])
DEFAULT_TARGET_SECONDS = float(_AUDIO_DEFAULTS["TARGET_SECONDS"])
DEFAULT_MONO = bool(_AUDIO_DEFAULTS["MONO"])
DEFAULT_NORMALIZATION = str(_AUDIO_DEFAULTS["NORMALIZATION"])
DEFAULT_EPS = float(_AUDIO_DEFAULTS["EPS"])
DEFAULT_DETERMINISTIC_CROP = str(_AUDIO_DEFAULTS["CROP_MODE"])
DEFAULT_FIXED_CROP_START = int(_AUDIO_DEFAULTS["FIXED_CROP_START"])
DEFAULT_RANDOM_CROP_GLOBAL_SEED = int(_AUDIO_DEFAULTS["RANDOM_CROP_GLOBAL_SEED"])
DEFAULT_INCLUDE_AUDIO_STATS = bool(_AUDIO_DEFAULTS["INCLUDE_AUDIO_STATS_DEFAULT"])
TARGET_SAMPLES_RULE = str(_AUDIO_DEFAULTS["TARGET_SAMPLES_RULE"])
PADDING_VALUE = float(_AUDIO_DEFAULTS["PADDING_VALUE"])
PADDING_MASK_TRUE_MEANS = str(_AUDIO_DEFAULTS["PADDING_MASK_TRUE_MEANS"])

if TARGET_SAMPLES_RULE != "round":
    raise ValueError(f"AUDIO.TARGET_SAMPLES_RULE must be 'round', got {TARGET_SAMPLES_RULE!r}")
if not DEFAULT_MONO:
    raise ValueError("AUDIO.MONO must remain True; this project only accepts mono waveform output")
if PADDING_VALUE != 0.0:
    raise ValueError(f"AUDIO.PADDING_VALUE must remain 0.0, got {PADDING_VALUE!r}")
if PADDING_MASK_TRUE_MEANS != "padding":
    raise ValueError(
        "AUDIO.PADDING_MASK_TRUE_MEANS must remain 'padding', "
        f"got {PADDING_MASK_TRUE_MEANS!r}"
    )

TRAIN_MODE = "train"
DETERMINISTIC_MODES = {
    "eval",
    "test",
    "export",
    "geometry",
    "quant_injection",
    "calibration",
    "observer",
}
MODE_ALIASES = {
    "val": "eval",
    "valid": "eval",
    "validation": "eval",
    "quant": "quant_injection",
    "quant_injection": "quant_injection",
    "quantinjection": "quant_injection",
    "quant-injection": "quant_injection",
    "quant injection": "quant_injection",
    "quant_calibration": "calibration",
}

OUTPUT_KEYS = (
    "sample_id",
    "waveform",
    "padding_mask",
    "label_id",
    "audio_path",
    "orig_sample_rate",
    "orig_num_samples",
    "valid_num_samples",
    "target_num_samples",
    "crop_start",
    "crop_end",
    "crop_mode",
    "normalization",
    "is_silent_or_near_silent",
    "audio_stats",
)
LIGHTWEIGHT_OUTPUT_KEYS = tuple(key for key in OUTPUT_KEYS if key != "audio_stats")

RESAMPLER_CACHE_MAX_SIZE = int(_AUDIO_DEFAULTS["RESAMPLER_CACHE_MAX_SIZE"])
DETERMINISTIC_CROP_START_CACHE_MAX_SIZE = int(_AUDIO_DEFAULTS["DETERMINISTIC_CROP_START_CACHE_MAX_SIZE"])
if RESAMPLER_CACHE_MAX_SIZE <= 0:
    raise ValueError(f"AUDIO.RESAMPLER_CACHE_MAX_SIZE must be positive, got {RESAMPLER_CACHE_MAX_SIZE}")
if DETERMINISTIC_CROP_START_CACHE_MAX_SIZE <= 0:
    raise ValueError(
        "AUDIO.DETERMINISTIC_CROP_START_CACHE_MAX_SIZE must be positive, "
        f"got {DETERMINISTIC_CROP_START_CACHE_MAX_SIZE}"
    )

_RESAMPLER_CACHE: OrderedDict[tuple[int, int], Any] = OrderedDict()
_DETERMINISTIC_CROP_START_CACHE: OrderedDict[tuple[str, int, int, str, int], int] = OrderedDict()
_CACHE_LOCK = Lock()


def _expected_output_keys(*, include_audio_stats: bool) -> tuple[str, ...]:
    return OUTPUT_KEYS if include_audio_stats else LIGHTWEIGHT_OUTPUT_KEYS


def _validate_output_contract(
    payload: Mapping[str, Any],
    *,
    include_audio_stats: bool,
    source: str,
) -> None:
    expected_keys = set(_expected_output_keys(include_audio_stats=include_audio_stats))
    observed_keys = set(payload.keys())
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            f"{source} violates audio input output contract: "
            f"missing={missing}, extra={extra}, "
            f"expected={list(_expected_output_keys(include_audio_stats=include_audio_stats))}"
        )


class AudioInputError(RuntimeError):
    """Raised when unified waveform input cannot be produced."""

    def __init__(
        self,
        message: str,
        *,
        sample_id: str | None = None,
        audio_path: str | Path | None = None,
    ) -> None:
        context = []
        if sample_id is not None:
            context.append(f"sample_id={sample_id!r}")
        if audio_path is not None:
            context.append(f"audio_path={str(audio_path)!r}")
        prefix = f"AudioInputError({', '.join(context)}): " if context else "AudioInputError: "
        super().__init__(prefix + message)
        self.sample_id = sample_id
        self.audio_path = str(audio_path) if audio_path is not None else None


@dataclass(frozen=True)
class AudioInputConfig:
    """Project-wide waveform input protocol.

    This module only produces mono waveform tensors and padding masks. It does
    not create mel features, spectrograms, BEATs inputs, logits, or metrics.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    target_seconds: float = DEFAULT_TARGET_SECONDS
    normalization: Literal["peak"] = DEFAULT_NORMALIZATION  # type: ignore[assignment]
    eps: float = DEFAULT_EPS
    deterministic_crop: Literal["center", "start", "fixed"] = DEFAULT_DETERMINISTIC_CROP  # type: ignore[assignment]
    fixed_crop_start: int = DEFAULT_FIXED_CROP_START

    def __post_init__(self) -> None:
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if float(self.target_seconds) <= 0:
            raise ValueError(f"target_seconds must be positive, got {self.target_seconds}")
        if self.normalization != "peak":
            raise ValueError("Only peak normalization is allowed by the audio input protocol.")
        if float(self.eps) <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.deterministic_crop not in {"center", "start", "fixed"}:
            raise ValueError(
                "deterministic_crop must be one of {'center', 'start', 'fixed'}, "
                f"got {self.deterministic_crop!r}"
            )
        if int(self.fixed_crop_start) < 0:
            raise ValueError(f"fixed_crop_start must be >= 0, got {self.fixed_crop_start}")

    @property
    def target_num_samples(self) -> int:
        target = int(round(int(self.sample_rate) * float(self.target_seconds)))
        if target <= 0:
            raise ValueError(
                "target_num_samples must be positive after applying "
                f"{TARGET_SAMPLES_RULE!r}, got {target}"
            )
        return target


@dataclass(frozen=True)
class AudioInput:
    sample_id: str
    waveform: torch.Tensor
    padding_mask: torch.Tensor
    label_id: int | None
    audio_path: str
    orig_sample_rate: int
    orig_num_samples: int
    valid_num_samples: int
    target_num_samples: int
    crop_start: int
    crop_end: int
    crop_mode: str
    normalization: str
    is_silent_or_near_silent: bool
    audio_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_audio_stats: bool = DEFAULT_INCLUDE_AUDIO_STATS) -> dict[str, Any]:
        result = {
            "sample_id": self.sample_id,
            "waveform": self.waveform,
            "padding_mask": self.padding_mask,
            "label_id": self.label_id,
            "audio_path": self.audio_path,
            "orig_sample_rate": self.orig_sample_rate,
            "orig_num_samples": self.orig_num_samples,
            "valid_num_samples": self.valid_num_samples,
            "target_num_samples": self.target_num_samples,
            "crop_start": self.crop_start,
            "crop_end": self.crop_end,
            "crop_mode": self.crop_mode,
            "normalization": self.normalization,
            "is_silent_or_near_silent": self.is_silent_or_near_silent,
        }
        if include_audio_stats:
            result["audio_stats"] = dict(self.audio_stats)
        _validate_output_contract(
            result,
            include_audio_stats=include_audio_stats,
            source="AudioInput.to_dict",
        )
        return result


def normalize_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    key = MODE_ALIASES.get(key, key)
    if key == TRAIN_MODE or key in DETERMINISTIC_MODES:
        return key
    raise ValueError(
        f"Unsupported audio input mode={mode!r}. "
        f"Use 'train' or one of {sorted(DETERMINISTIC_MODES)}."
    )


def stable_uint64_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _lru_get_or_create(
    cache: OrderedDict[Any, Any],
    key: Any,
    create_value: Callable[[], Any],
    *,
    max_size: int,
) -> tuple[Any, bool]:
    # 轻量 LRU：同一进程内复用昂贵对象，避免每个样本重复构造。
    with _CACHE_LOCK:
        if key in cache:
            value = cache[key]
            cache.move_to_end(key)
            return value, True

    value = create_value()

    with _CACHE_LOCK:
        if key in cache:
            cached_value = cache[key]
            cache.move_to_end(key)
            return cached_value, True

        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > int(max_size):
            cache.popitem(last=False)
        return value, False


def _get_cached_torchaudio_resampler(
    *,
    orig_sample_rate: int,
    target_sample_rate: int,
) -> tuple[Any, bool]:
    key = (int(orig_sample_rate), int(target_sample_rate))

    def create_resampler() -> Any:
        import torchaudio.transforms as T

        return T.Resample(
            orig_freq=int(orig_sample_rate),
            new_freq=int(target_sample_rate),
        )

    return _lru_get_or_create(
        _RESAMPLER_CACHE,
        key,
        create_resampler,
        max_size=RESAMPLER_CACHE_MAX_SIZE,
    )


def _deterministic_crop_start_cache_key(
    *,
    sample_id: str,
    num_samples: int,
    target_num_samples: int,
    config: AudioInputConfig,
) -> tuple[str, int, int, str, int]:
    return (
        str(sample_id),
        int(num_samples),
        int(target_num_samples),
        str(config.deterministic_crop),
        int(config.fixed_crop_start),
    )


def _get_cached_deterministic_crop_start(
    *,
    sample_id: str,
    num_samples: int,
    target_num_samples: int,
    config: AudioInputConfig,
    create_value: Callable[[], int],
) -> tuple[int, bool]:
    key = _deterministic_crop_start_cache_key(
        sample_id=sample_id,
        num_samples=int(num_samples),
        target_num_samples=int(target_num_samples),
        config=config,
    )
    value, cache_hit = _lru_get_or_create(
        _DETERMINISTIC_CROP_START_CACHE,
        key,
        create_value,
        max_size=DETERMINISTIC_CROP_START_CACHE_MAX_SIZE,
    )
    return int(value), bool(cache_hit)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "<na>"}


def _text_value(value: Any) -> str | None:
    if _missing(value):
        return None
    return str(value).strip()


def _mapping_get(row: Mapping[str, Any], key: str) -> Any:
    try:
        return row[key]
    except KeyError:
        return None


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = _text_value(_mapping_get(row, key))
    if value is None:
        raise AudioInputError(f"missing required column {key!r}")
    return value


def _parse_label_id(
    value: Any,
    *,
    sample_id: str,
    audio_path: str | Path | None,
    require_label_id: bool,
) -> int | None:
    if _missing(value):
        if require_label_id:
            raise AudioInputError(
                "missing required label_id; this module never builds label mappings",
                sample_id=sample_id,
                audio_path=audio_path,
            )
        return None
    if isinstance(value, bool):
        raise AudioInputError(
            f"invalid label_id={value!r}",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise AudioInputError(
                f"invalid label_id={value!r}",
                sample_id=sample_id,
                audio_path=audio_path,
            )
        return int(value)
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise AudioInputError(
            f"invalid label_id={value!r}",
            sample_id=sample_id,
            audio_path=audio_path,
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise AudioInputError(
            f"invalid label_id={value!r}",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    return int(numeric)


def infer_dataset_root_from_split_csv(split_csv: str | Path) -> Path:
    split_csv = _portable_path(split_csv)
    if split_csv.parent.name.lower() == "splits":
        return split_csv.parent.parent
    return split_csv.parent


def _portable_path_text(value: str | Path) -> str:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _portable_path(value: str | Path) -> Path:
    return Path(_portable_path_text(value))


def _looks_like_windows_absolute(text: str) -> bool:
    return len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] == "/"


def _windows_uploaded_local_data_candidate(text: str, dataset_root: Path | None) -> Path | None:
    marker = "local_data/"
    if dataset_root is None or marker not in text:
        return None
    return dataset_root / text.split(marker, 1)[1]


def resolve_audio_path_from_row(
    row: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    roots = []
    dataset_root_path = _portable_path(dataset_root) if dataset_root is not None else None
    project_root_path = _portable_path(project_root) if project_root is not None else None
    if dataset_root is not None:
        roots.append(dataset_root_path)
    if project_root is not None:
        roots.append(project_root_path)

    candidates: list[Path] = []

    def add_candidate(raw_value: Any, preferred_roots: Sequence[Path]) -> None:
        text = _text_value(raw_value)
        if text is None:
            return
        text = _portable_path_text(text)
        path = Path(text)
        if path.is_absolute():
            candidates.append(path)
            return
        if _looks_like_windows_absolute(text):
            relocated = _windows_uploaded_local_data_candidate(text, dataset_root_path)
            if relocated is not None:
                candidates.append(relocated)
            if project_root_path is not None:
                project_marker = f"/{project_root_path.name}/"
                if project_marker in text:
                    candidates.append(project_root_path / text.split(project_marker, 1)[1])
            candidates.append(path)
            return
        if text.startswith("local_data/") and project_root_path is not None:
            candidates.append(project_root_path / path)
        for root in preferred_roots:
            candidates.append(root / path)
        candidates.append(path)

    for column in ("audio_path", "path", "absolute_path"):
        add_candidate(_mapping_get(row, column), roots)
    add_candidate(_mapping_get(row, "filepath"), [project_root_path] if project_root_path is not None else roots)
    add_candidate(_mapping_get(row, "file_id"), [project_root_path] if project_root_path is not None else roots)
    add_candidate(_mapping_get(row, "rel_path"), [dataset_root_path] if dataset_root_path is not None else roots)

    deduped: list[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return candidate

    sample_id = _text_value(_mapping_get(row, "sample_id"))
    attempted = [str(path) for path in deduped]
    raise AudioInputError(
        f"audio file not found; attempted paths={attempted}",
        sample_id=sample_id,
        audio_path=attempted[0] if attempted else None,
    )


def _validate_non_empty_finite(
    waveform: torch.Tensor,
    *,
    stage: str,
    sample_id: str,
    audio_path: str | Path,
) -> None:
    if waveform.numel() == 0:
        raise AudioInputError(
            f"{stage} waveform is empty",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    if waveform.shape[-1] <= 0:
        raise AudioInputError(
            f"{stage} waveform has zero time samples",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    if not torch.isfinite(waveform).all().item():
        raise AudioInputError(
            f"{stage} waveform contains NaN or Inf",
            sample_id=sample_id,
            audio_path=audio_path,
        )


def _read_audio(
    audio_path: str | Path,
    *,
    sample_id: str,
) -> tuple[torch.Tensor, int, str]:
    path = _portable_path(audio_path)
    if not path.exists():
        raise AudioInputError(
            "audio file does not exist",
            sample_id=sample_id,
            audio_path=path,
        )

    torchaudio_error: Exception | None = None
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.detach().cpu().to(dtype=torch.float32)
        return waveform, int(sample_rate), "torchaudio"
    except Exception as exc:  # pragma: no cover - backend fallback depends on host codecs.
        torchaudio_error = exc

    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy()).to(dtype=torch.float32)
        return waveform, int(sample_rate), "soundfile"
    except Exception as exc:
        raise AudioInputError(
            "audio read failed with torchaudio and soundfile; "
            f"torchaudio_error={torchaudio_error!r}; soundfile_error={exc!r}",
            sample_id=sample_id,
            audio_path=path,
        ) from exc


def _as_channels_first(
    waveform: torch.Tensor,
    *,
    sample_id: str,
    audio_path: str | Path,
) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise AudioInputError(
            f"expected raw audio shape [channels, samples] or [samples], got {tuple(waveform.shape)}",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    if waveform.shape[0] <= 0:
        raise AudioInputError(
            "raw audio has zero channels",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    return waveform


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] == 1:
        return waveform[0].contiguous()
    return waveform.mean(dim=0).contiguous()


def _resample_mono(
    waveform: torch.Tensor,
    *,
    orig_sample_rate: int,
    target_sample_rate: int,
    sample_id: str,
    audio_path: str | Path,
) -> tuple[torch.Tensor, str]:
    if int(orig_sample_rate) <= 0:
        raise AudioInputError(
            f"invalid original sample rate={orig_sample_rate}",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    if int(target_sample_rate) <= 0:
        raise AudioInputError(
            f"invalid target sample rate={target_sample_rate}",
            sample_id=sample_id,
            audio_path=audio_path,
        )
    if int(orig_sample_rate) == int(target_sample_rate):
        return waveform.to(dtype=torch.float32).contiguous(), "identity"

    torchaudio_error: Exception | None = None
    try:
        resampler, cache_hit = _get_cached_torchaudio_resampler(
            orig_sample_rate=int(orig_sample_rate),
            target_sample_rate=int(target_sample_rate),
        )
        resampled = resampler(waveform.unsqueeze(0)).squeeze(0)
        cache_state = "hit" if cache_hit else "miss"
        return resampled.to(dtype=torch.float32).contiguous(), f"torchaudio_resample_lru_{cache_state}"
    except Exception as exc:  # pragma: no cover - backend fallback depends on host install.
        torchaudio_error = exc

    try:
        import numpy as np
        from scipy.signal import resample_poly

        divisor = math.gcd(int(orig_sample_rate), int(target_sample_rate))
        up = int(target_sample_rate) // divisor
        down = int(orig_sample_rate) // divisor
        array = waveform.detach().cpu().numpy()
        resampled_np = resample_poly(array, up=up, down=down).astype(np.float32, copy=False)
        return torch.from_numpy(resampled_np.copy()).to(dtype=torch.float32).contiguous(), "scipy"
    except Exception as exc:
        raise AudioInputError(
            "resampling failed with torchaudio and scipy; "
            f"torchaudio_error={torchaudio_error!r}; scipy_error={exc!r}",
            sample_id=sample_id,
            audio_path=audio_path,
        ) from exc


def _rms(waveform: torch.Tensor) -> float:
    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.float().square())).item())


def _peak_normalize(
    waveform: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, float, float, bool, bool]:
    peak = float(waveform.abs().max().item())
    rms = _rms(waveform)
    is_silent_or_near_silent = peak < float(eps)
    if is_silent_or_near_silent:
        return waveform.to(dtype=torch.float32).contiguous(), peak, rms, True, False
    normalized = torch.clamp(waveform / peak, min=-1.0, max=1.0)
    return normalized.to(dtype=torch.float32).contiguous(), peak, rms, False, True


def _crop_mode_for(normalized_mode: str, config: AudioInputConfig) -> str:
    if normalized_mode == TRAIN_MODE:
        return "random"
    return str(config.deterministic_crop)


def _crop_start_for(
    *,
    num_samples: int,
    target_num_samples: int,
    sample_id: str,
    normalized_mode: str,
    global_seed: int,
    epoch: int,
    config: AudioInputConfig,
) -> tuple[int, str]:
    def deterministic_crop_start() -> int:
        if num_samples <= target_num_samples:
            return 0

        max_start = int(num_samples) - int(target_num_samples)
        if config.deterministic_crop == "center":
            return int(max_start // 2)
        if config.deterministic_crop == "start":
            return 0
        if config.deterministic_crop == "fixed":
            return min(int(config.fixed_crop_start), max_start)
        raise AssertionError(f"unexpected deterministic_crop={config.deterministic_crop!r}")

    if normalized_mode != TRAIN_MODE:
        crop_start, cache_hit = _get_cached_deterministic_crop_start(
            sample_id=sample_id,
            num_samples=int(num_samples),
            target_num_samples=int(target_num_samples),
            config=config,
            create_value=deterministic_crop_start,
        )
        cache_state = "deterministic_lru_hit" if cache_hit else "deterministic_lru_miss"
        return crop_start, cache_state

    if num_samples <= target_num_samples:
        return 0, "train_no_crop"

    max_start = int(num_samples) - int(target_num_samples)
    seed = stable_uint64_seed(
        "audio_input_random_crop",
        int(global_seed),
        int(epoch),
        str(sample_id),
    )
    return int(seed % (max_start + 1)), "train_random"


def _crop_or_pad(
    waveform: torch.Tensor,
    *,
    sample_id: str,
    normalized_mode: str,
    global_seed: int,
    epoch: int,
    config: AudioInputConfig,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, str, str, bool, bool]:
    target_num_samples = int(config.target_num_samples)
    num_samples = int(waveform.numel())
    crop_mode = _crop_mode_for(normalized_mode, config)
    crop_start, crop_start_cache = _crop_start_for(
        num_samples=num_samples,
        target_num_samples=target_num_samples,
        sample_id=sample_id,
        normalized_mode=normalized_mode,
        global_seed=int(global_seed),
        epoch=int(epoch),
        config=config,
    )

    if num_samples > target_num_samples:
        crop_end = crop_start + target_num_samples
        fixed = waveform[crop_start:crop_end].contiguous()
        padding_mask = torch.zeros(target_num_samples, dtype=torch.bool)
        valid_num_samples = target_num_samples
        return (
            fixed.to(dtype=torch.float32),
            padding_mask,
            valid_num_samples,
            crop_start,
            crop_end,
            crop_mode,
            crop_start_cache,
            False,
            True,
        )

    valid_num_samples = num_samples
    crop_end = num_samples
    fixed = torch.full((target_num_samples,), float(PADDING_VALUE), dtype=torch.float32)
    fixed[:valid_num_samples] = waveform.to(dtype=torch.float32)
    padding_mask = torch.zeros(target_num_samples, dtype=torch.bool)
    if valid_num_samples < target_num_samples:
        padding_mask[valid_num_samples:] = True

    return (
        fixed,
        padding_mask,
        valid_num_samples,
        crop_start,
        crop_end,
        crop_mode,
        crop_start_cache,
        bool(valid_num_samples < target_num_samples),
        False,
    )


def load_audio_input(
    *,
    audio_path: str | Path,
    sample_id: str,
    label_id: int | None,
    mode: str,
    audio_config: AudioInputConfig | None = None,
    global_seed: int = DEFAULT_RANDOM_CROP_GLOBAL_SEED,
    epoch: int = 0,
) -> AudioInput:
    config = audio_config or AudioInputConfig()
    normalized_mode = normalize_mode(mode)
    path = _portable_path(audio_path)
    sample_id = str(sample_id)

    try:
        raw_waveform, orig_sample_rate, read_backend = _read_audio(path, sample_id=sample_id)
        raw_waveform = _as_channels_first(raw_waveform, sample_id=sample_id, audio_path=path)
        _validate_non_empty_finite(
            raw_waveform,
            stage="raw",
            sample_id=sample_id,
            audio_path=path,
        )

        raw_num_channels = int(raw_waveform.shape[0])
        orig_num_samples = int(raw_waveform.shape[-1])
        mono = _to_mono(raw_waveform)
        _validate_non_empty_finite(
            mono,
            stage="mono",
            sample_id=sample_id,
            audio_path=path,
        )

        resampled, resample_backend = _resample_mono(
            mono,
            orig_sample_rate=int(orig_sample_rate),
            target_sample_rate=int(config.sample_rate),
            sample_id=sample_id,
            audio_path=path,
        )
        _validate_non_empty_finite(
            resampled,
            stage="resampled",
            sample_id=sample_id,
            audio_path=path,
        )

        resampled_num_samples = int(resampled.numel())
        normalized, pre_peak, pre_rms, is_silent, normalization_applied = _peak_normalize(
            resampled,
            eps=float(config.eps),
        )
        _validate_non_empty_finite(
            normalized,
            stage="normalized",
            sample_id=sample_id,
            audio_path=path,
        )

        (
            fixed_waveform,
            padding_mask,
            valid_num_samples,
            crop_start,
            crop_end,
            crop_mode,
            crop_start_cache,
            is_padded,
            is_cropped,
        ) = _crop_or_pad(
            normalized,
            sample_id=sample_id,
            normalized_mode=normalized_mode,
            global_seed=int(global_seed),
            epoch=int(epoch),
            config=config,
        )
        _validate_non_empty_finite(
            fixed_waveform,
            stage="final",
            sample_id=sample_id,
            audio_path=path,
        )

        target_num_samples = int(config.target_num_samples)
        if tuple(fixed_waveform.shape) != (target_num_samples,):
            raise AudioInputError(
                f"final waveform shape must be [{target_num_samples}], got {tuple(fixed_waveform.shape)}",
                sample_id=sample_id,
                audio_path=path,
            )
        if tuple(padding_mask.shape) != (target_num_samples,):
            raise AudioInputError(
                f"padding_mask shape must be [{target_num_samples}], got {tuple(padding_mask.shape)}",
                sample_id=sample_id,
                audio_path=path,
            )
        if padding_mask.dtype != torch.bool:
            raise AudioInputError(
                f"padding_mask dtype must be torch.bool, got {padding_mask.dtype}",
                sample_id=sample_id,
                audio_path=path,
            )

        padding_num_samples = int(padding_mask.sum().item())
        # padding 始终是右侧连续 0，因此有效区域可直接切片，避免布尔索引额外分配。
        valid_region = fixed_waveform[: int(valid_num_samples)]
        post_peak = float(normalized.abs().max().item())
        post_rms = _rms(normalized)
        final_peak = float(fixed_waveform.abs().max().item())
        final_rms = _rms(fixed_waveform)
        valid_rms = _rms(valid_region) if valid_region.numel() > 0 else 0.0

        stats = {
            "mode": normalized_mode,
            "read_backend": read_backend,
            "resample_backend": resample_backend,
            "raw_num_channels": raw_num_channels,
            "orig_sample_rate": int(orig_sample_rate),
            "orig_num_samples": orig_num_samples,
            "orig_duration_seconds": float(orig_num_samples / float(orig_sample_rate)),
            "target_sample_rate": int(config.sample_rate),
            "target_seconds": float(config.target_seconds),
            "target_num_samples": target_num_samples,
            "target_samples_rule": TARGET_SAMPLES_RULE,
            "resampled_num_samples": resampled_num_samples,
            "resampled_duration_seconds": float(resampled_num_samples / float(config.sample_rate)),
            "pre_normalization_peak": pre_peak,
            "pre_normalization_rms": pre_rms,
            "post_normalization_peak": post_peak,
            "post_normalization_rms": post_rms,
            "final_peak": final_peak,
            "final_rms": final_rms,
            "valid_rms": valid_rms,
            "normalization": config.normalization,
            "normalization_eps": float(config.eps),
            "normalization_applied": bool(normalization_applied),
            "is_silent_or_near_silent": bool(is_silent),
            "crop_mode": crop_mode,
            "crop_start": int(crop_start),
            "crop_end": int(crop_end),
            "crop_start_cache": crop_start_cache,
            "is_cropped": bool(is_cropped),
            "is_padded": bool(is_padded),
            "padding_num_samples": padding_num_samples,
            "padding_ratio": float(padding_num_samples / float(target_num_samples)),
            "padding_mask_true_means": PADDING_MASK_TRUE_MEANS,
            "valid_region_rule": "fixed_waveform[:valid_num_samples]",
            "valid_num_samples": int(valid_num_samples),
            "global_seed": int(global_seed),
            "epoch": int(epoch),
        }

        return AudioInput(
            sample_id=sample_id,
            waveform=fixed_waveform.to(dtype=torch.float32).contiguous(),
            padding_mask=padding_mask.contiguous(),
            label_id=label_id,
            audio_path=str(path),
            orig_sample_rate=int(orig_sample_rate),
            orig_num_samples=orig_num_samples,
            valid_num_samples=int(valid_num_samples),
            target_num_samples=target_num_samples,
            crop_start=int(crop_start),
            crop_end=int(crop_end),
            crop_mode=crop_mode,
            normalization=config.normalization,
            is_silent_or_near_silent=bool(is_silent),
            audio_stats=stats,
        )
    except AudioInputError:
        raise
    except Exception as exc:
        raise AudioInputError(
            f"failed to build unified audio input: {exc}",
            sample_id=sample_id,
            audio_path=path,
        ) from exc


def load_audio_input_from_row(
    row: Mapping[str, Any],
    *,
    mode: str,
    dataset_root: str | Path | None = None,
    project_root: str | Path | None = None,
    audio_config: AudioInputConfig | None = None,
    global_seed: int = DEFAULT_RANDOM_CROP_GLOBAL_SEED,
    epoch: int = 0,
    require_label_id: bool = True,
) -> AudioInput:
    sample_id = _required_text(row, "sample_id")
    audio_path = resolve_audio_path_from_row(
        row,
        dataset_root=dataset_root,
        project_root=project_root,
    )
    label_id = _parse_label_id(
        _mapping_get(row, "label_id"),
        sample_id=sample_id,
        audio_path=audio_path,
        require_label_id=bool(require_label_id),
    )
    return load_audio_input(
        audio_path=audio_path,
        sample_id=sample_id,
        label_id=label_id,
        mode=mode,
        audio_config=audio_config,
        global_seed=int(global_seed),
        epoch=int(epoch),
    )


class AudioInputDataset(Dataset):
    """CSV-backed dataset that applies the unified waveform protocol."""

    def __init__(
        self,
        split_csv: str | Path,
        *,
        mode: str,
        dataset_root: str | Path | None = None,
        project_root: str | Path | None = None,
        audio_config: AudioInputConfig | None = None,
        global_seed: int = DEFAULT_RANDOM_CROP_GLOBAL_SEED,
        epoch: int = 0,
        return_dict: bool = True,
        require_label_id: bool = True,
        include_audio_stats: bool = DEFAULT_INCLUDE_AUDIO_STATS,
    ) -> None:
        import pandas as pd

        self.split_csv = _portable_path(split_csv)
        if not self.split_csv.exists():
            raise FileNotFoundError(f"split_csv does not exist: {self.split_csv}")
        self.frame = pd.read_csv(self.split_csv)
        if self.frame.empty:
            raise ValueError(f"split_csv is empty: {self.split_csv}")

        self.mode = normalize_mode(mode)
        self.dataset_root = _portable_path(dataset_root) if dataset_root is not None else infer_dataset_root_from_split_csv(self.split_csv)
        self.project_root = _portable_path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
        self.audio_config = audio_config or AudioInputConfig()
        self.global_seed = int(global_seed)
        self.epoch = int(epoch)
        self.return_dict = bool(return_dict)
        self.require_label_id = bool(require_label_id)
        self.include_audio_stats = bool(include_audio_stats)

    def __len__(self) -> int:
        return int(len(self.frame))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> AudioInput | dict[str, Any]:
        row = self.frame.iloc[int(index)]
        item = load_audio_input_from_row(
            row,
            mode=self.mode,
            dataset_root=self.dataset_root,
            project_root=self.project_root,
            audio_config=self.audio_config,
            global_seed=self.global_seed,
            epoch=self.epoch,
            require_label_id=self.require_label_id,
        )
        if self.return_dict:
            return item.to_dict(include_audio_stats=self.include_audio_stats)
        return item


def _item_to_dict(
    item: AudioInput | Mapping[str, Any],
    *,
    include_audio_stats: bool | None = None,
) -> dict[str, Any]:
    if isinstance(item, AudioInput):
        return item.to_dict(include_audio_stats=include_audio_stats is True)
    if isinstance(item, Mapping):
        result = dict(item)
        if include_audio_stats is False:
            result.pop("audio_stats", None)
        item_has_audio_stats = "audio_stats" in result
        if include_audio_stats is True and not item_has_audio_stats:
            raise ValueError("include_audio_stats=True but item does not contain audio_stats")
        _validate_output_contract(
            result,
            include_audio_stats=item_has_audio_stats,
            source="audio_input_collate_fn item",
        )
        return result
    raise TypeError(f"audio_input_collate_fn expected AudioInput or Mapping, got {type(item)!r}")


def audio_input_collate_fn(
    batch: Sequence[AudioInput | Mapping[str, Any]],
    *,
    include_audio_stats: bool | None = None,
) -> dict[str, Any]:
    items = [_item_to_dict(item, include_audio_stats=include_audio_stats) for item in batch]
    if not items:
        raise ValueError("audio_input_collate_fn received an empty batch")

    output: dict[str, Any] = {}
    output["sample_id"] = [item["sample_id"] for item in items]
    output["waveform"] = torch.stack([item["waveform"] for item in items], dim=0).to(dtype=torch.float32)
    output["padding_mask"] = torch.stack([item["padding_mask"] for item in items], dim=0).to(dtype=torch.bool)
    output["label_id"] = (
        torch.tensor([item["label_id"] for item in items], dtype=torch.long)
        if all(item["label_id"] is not None for item in items)
        else [item["label_id"] for item in items]
    )
    output["audio_path"] = [item["audio_path"] for item in items]

    int_fields = (
        "orig_sample_rate",
        "orig_num_samples",
        "valid_num_samples",
        "target_num_samples",
        "crop_start",
        "crop_end",
    )
    for field_name in int_fields:
        output[field_name] = torch.tensor([int(item[field_name]) for item in items], dtype=torch.long)

    output["crop_mode"] = [item["crop_mode"] for item in items]
    output["normalization"] = [item["normalization"] for item in items]
    output["is_silent_or_near_silent"] = torch.tensor(
        [bool(item["is_silent_or_near_silent"]) for item in items],
        dtype=torch.bool,
    )

    emit_audio_stats = (
        include_audio_stats is True
        or (include_audio_stats is None and all("audio_stats" in item for item in items))
    )
    if emit_audio_stats:
        missing_indices = [idx for idx, item in enumerate(items) if "audio_stats" not in item]
        if missing_indices:
            raise ValueError(
                "include_audio_stats=True but batch items do not contain audio_stats "
                f"at indices={missing_indices[:8]}"
        )
        output["audio_stats"] = [item["audio_stats"] for item in items]
    else:
        extra_stats_indices = [idx for idx, item in enumerate(items) if "audio_stats" in item]
        if extra_stats_indices:
            raise ValueError(
                "audio_input_collate_fn received mixed lightweight/full items; "
                "pass include_audio_stats=True or make all items lightweight. "
                f"full item indices={extra_stats_indices[:8]}"
            )

    target_num_samples = int(output["target_num_samples"][0].item())
    if tuple(output["waveform"].shape[1:]) != (target_num_samples,):
        raise ValueError(
            "collated waveform must have shape [B, target_num_samples], "
            f"got {tuple(output['waveform'].shape)}"
        )
    if tuple(output["padding_mask"].shape) != tuple(output["waveform"].shape):
        raise ValueError(
            "collated padding_mask must match waveform shape, "
            f"got {tuple(output['padding_mask'].shape)} vs {tuple(output['waveform'].shape)}"
        )
    _validate_output_contract(
        output,
        include_audio_stats=emit_audio_stats,
        source="audio_input_collate_fn output",
    )
    return output


__all__ = [
    "AudioInput",
    "AudioInputConfig",
    "AudioInputDataset",
    "AudioInputError",
    "DEFAULT_DETERMINISTIC_CROP",
    "DETERMINISTIC_CROP_START_CACHE_MAX_SIZE",
    "DEFAULT_EPS",
    "DEFAULT_FIXED_CROP_START",
    "DEFAULT_INCLUDE_AUDIO_STATS",
    "DEFAULT_MONO",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_RANDOM_CROP_GLOBAL_SEED",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_TARGET_SECONDS",
    "LIGHTWEIGHT_OUTPUT_KEYS",
    "OUTPUT_KEYS",
    "PADDING_MASK_TRUE_MEANS",
    "PADDING_VALUE",
    "RESAMPLER_CACHE_MAX_SIZE",
    "TARGET_SAMPLES_RULE",
    "audio_input_collate_fn",
    "infer_dataset_root_from_split_csv",
    "load_audio_input",
    "load_audio_input_from_row",
    "normalize_mode",
    "resolve_audio_path_from_row",
    "stable_uint64_seed",
]
