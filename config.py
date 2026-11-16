import os


def _sanitize_positive_int_env(name: str, default: int) -> None:
    value = os.environ.get(name)
    try:
        if value is not None and int(value) > 0:
            return
    except ValueError:
        pass
    os.environ[name] = str(int(default))


_sanitize_positive_int_env("OMP_NUM_THREADS", 4)
_sanitize_positive_int_env("MKL_NUM_THREADS", 4)

import random
import numpy as np
import torch

from pathlib import Path

# --- [环境设置] ---
BASE_DIR: Path = Path(__file__).resolve().parent

BASE_RESULT_DIR: Path = BASE_DIR / "outputs"


SPLIT_SEED = 42
DEFAULT_SEEDS = (42, 3407, 2026)
TRAINING_SEEDS = DEFAULT_SEEDS

# =========================================================
# Device Protocol
# =========================================================
DEVICE = os.environ.get("LQD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if str(DEVICE).startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"DEVICE={DEVICE!r} 需要 CUDA，但当前环境 torch.cuda.is_available() 为 False")

# =========================================================
# Global Audio Protocol
# =========================================================
AUDIO = {
    "TARGET_SR": 16000,
    "TARGET_SECONDS": 10.0,
    "MONO": True,
    "NORMALIZATION": "peak",
    "EPS": 1e-8,
    "CROP_MODE": "center",
    "TARGET_SAMPLES_RULE": "round",
    "PADDING_VALUE": 0.0,
    "PADDING_MASK_TRUE_MEANS": "padding",
    "FIXED_CROP_START": 0,
    "RANDOM_CROP_GLOBAL_SEED": SPLIT_SEED,
    "INCLUDE_AUDIO_STATS_DEFAULT": False,
    "RESAMPLER_CACHE_MAX_SIZE": 16,
    "DETERMINISTIC_CROP_START_CACHE_MAX_SIZE": 131_072,
}

# OER top-k protocol:
# scalar OER uses flatten scalar oracle; token OER uses top-k token vectors.
OER_SCALAR_TOP_RATIO = 0.001
TOKEN_TOPK_RATIO = 0.05

# =========================================================
# W4A4 Fake Quantization
# =========================================================

QUANT = {
    "TARGET_SCOPE": "beats_transformer_ffn",
    "TARGET_LINEAR_NAMES": ("fc1", "fc2"),
    "DEFAULT_MODE": "w4a4",
    "WEIGHT": {
        "BITS": 4,
        "QMIN": -7,
        "QMAX": 7,
        "SYMMETRIC": True,
        "GRANULARITY": "per_channel",
        "CHANNEL_AXIS": 0,
        "EPS": 1e-8,
    },
    "ACTIVATION": {
        "BITS": 4,
        "QMIN": -7,
        "QMAX": 7,
        "SYMMETRIC": True,
        "GRANULARITY": "per_tensor",
        "CALIBRATION_METHOD": "percentile",
        "PERCENTILE_MODE": "retained_mass",
        "PERCENTILE": 99.9,
        "OBSERVER_BINS": 2048,
        "EPS": 1e-8,
    },
}

# =========================================================
# Observer Protocol
# =========================================================

OBSERVER = {
    "DEFAULT_PERCENTILES": (99.0, 99.9, 99.99),
    "PERCENTILE_BACKEND": "two_pass_histogram",
    "HISTOGRAM_BINS": QUANT["ACTIVATION"]["OBSERVER_BINS"],
    "ACTIVATION": {
        "GRANULARITY": QUANT["ACTIVATION"]["GRANULARITY"],
        "PERCENTILES": (99.0, QUANT["ACTIVATION"]["PERCENTILE"], 99.99),
        "PERCENTILE_LOW": None,
        "PERCENTILE_HIGH": QUANT["ACTIVATION"]["PERCENTILE"],
        "PERCENTILE_BACKEND": "two_pass_histogram",
        "HISTOGRAM_BINS": QUANT["ACTIVATION"]["OBSERVER_BINS"],
        "EXCLUDE_PADDING": True,
        "EPS": QUANT["ACTIVATION"]["EPS"],
    },
    "WEIGHT": {
        "GRANULARITY": QUANT["WEIGHT"]["GRANULARITY"],
        "CHANNEL_AXIS": QUANT["WEIGHT"]["CHANNEL_AXIS"],
        "PERCENTILES": (99.0, 99.9, 99.99),
        "PERCENTILE_LOW": None,
        "PERCENTILE_HIGH": 99.9,
        "PERCENTILE_BACKEND": "two_pass_histogram",
        "HISTOGRAM_BINS": QUANT["ACTIVATION"]["OBSERVER_BINS"],
        "EPS": QUANT["WEIGHT"]["EPS"],
    },
}

# =========================================================
# Common Metrics Protocol
# =========================================================

METRICS = {
    "EPS": 1e-12,
    "SCALAR_OER_TOP_RATIO": OER_SCALAR_TOP_RATIO,
    "MARGIN_MIN_ABS": 1e-6,
    "FISHER_NORMALIZE_BY_DOF": True,
    "GEOMETRY_VARIANCE_NORMALIZATION": "dof",
    "GEOMETRY_CENTER_SHIFT_MODE": "absolute",
    "SIGNED_INT4_SYMMETRIC_QMIN": QUANT["WEIGHT"]["QMIN"],
    "SIGNED_INT4_SYMMETRIC_QMAX": QUANT["WEIGHT"]["QMAX"],
}


ACTIVATION_DISTRIBUTION = {
    "LAYOUT_CHECK_TARGETS": [
        (2, "fc2"),
        (0, "fc1"),
    ],
    "CALIBRATION_SHUFFLE_SEED": SPLIT_SEED,
}

FFN_SELECTED_TARGET_MODULES = [
    "L1.fc1",
    "L3.fc1",
    "L4.fc1",
    "L9.fc1",
    "L0.fc2",
    "L3.fc2",
    "L9.fc2",
    "L11.fc2",
]

# =========================================================
# FFN Layer Heatmap
# =========================================================
FFN_LAYER_HEATMAP = {
    "DEVICE": DEVICE,
    "BATCH_SIZE": 8,
    "EVAL_BATCH_SIZE": 8,
    "CALIBRATION_BATCH_SIZE": 8,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
    "PERSISTENT_WORKERS": True,
    "PREFETCH_FACTOR": 2,
    "RESUME": True,
}

# =========================================================
# Semantic Geometry Projection Probe
# =========================================================
SEMANTIC_GEOMETRY_MODE_RUNNER = {
    "DEVICE": DEVICE,
    "BATCH_SIZE": 8,
    "EVAL_BATCH_SIZE": 8,
    "CALIBRATION_BATCH_SIZE": 8,
}

# =========================================================
# FFN Token Survival Probe
# =========================================================
FFN_TOKEN_SURVIVAL_PROBE = {
    "DEVICE": DEVICE,
    "EVAL_BATCH_SIZE": 8,
    "NUM_WORKERS": 8,
    "PIN_MEMORY": True,
    "PERSISTENT_WORKERS": True,
    "PREFETCH_FACTOR": 2,
    "BLOCK_TRACE_GROUP_SIZE": 6,
    "STREAMING_SUMMARY": True,
    "SUMMARY_READ_CHUNKSIZE": 50000,
    "WRITE_COMBINED_SAMPLE_CSV": False,
    "RUN_CASE_PLOTS": False,
    "EXPORT_DIAGNOSTIC_COMPARISON": False,
}
DATASETS = {
    "local_data": {
        "ROOT": BASE_DIR / "local_data",
        "DATA_DIR": BASE_DIR / "local_data" / "data",
        "SPLIT_DIR": BASE_DIR / "local_data" / "splits",
        "TRAIN_CSV": BASE_DIR / "local_data" / "splits" / "train.csv",
        "EVAL_CSV": BASE_DIR / "local_data" / "splits" / "eval.csv",
        "TEST_CSV": BASE_DIR / "local_data" / "splits" / "test.csv",
        "FULL_CSV": BASE_DIR / "local_data" / "splits" / "full_dataset.csv",
        "LABEL_MAPPING": BASE_DIR / "local_data" / "splits" / "label_mapping.json",
        "SPLIT_REPORT": BASE_DIR / "local_data" / "splits" / "split_report.json",
        "CALIBRATION_CSV": BASE_DIR / "local_data" / "splits" / "calibration.csv",
        "CALIBRATION_MANIFEST": BASE_DIR / "local_data" / "splits" / "calibration.manifest.json",
    },
    "PER_audio": {
        "ROOT": BASE_DIR / "PER_audio",
        "DATA_DIR": BASE_DIR / "PER_audio",
        "SPLIT_DIR": BASE_DIR / "PER_audio" / "splits",
        "TRAIN_CSV": BASE_DIR / "PER_audio" / "splits" / "train.csv",
        "EVAL_CSV": BASE_DIR / "PER_audio" / "splits" / "eval.csv",
        "TEST_CSV": BASE_DIR / "PER_audio" / "splits" / "test.csv",
        "FULL_CSV": BASE_DIR / "PER_audio" / "splits" / "full_dataset.csv",
        "LABEL_MAPPING": BASE_DIR / "PER_audio" / "splits" / "label_mapping.json",
        "SPLIT_REPORT": BASE_DIR / "PER_audio" / "splits" / "split_report.json",
        "CALIBRATION_CSV": BASE_DIR / "PER_audio" / "splits" / "calibration.csv",
        "CALIBRATION_MANIFEST": BASE_DIR / "PER_audio" / "splits" / "calibration.manifest.json",
    },
}

#ACTIVE_DATASET = os.environ.get("LQD_DATASET", "PER_audio")
ACTIVE_DATASET = os.environ.get("LQD_DATASET", "local_data")
if ACTIVE_DATASET not in DATASETS:
    raise ValueError(f"Unknown ACTIVE_DATASET={ACTIVE_DATASET!r}; choose one of {tuple(DATASETS)}")

CURRENT_DATASET = DATASETS[ACTIVE_DATASET]
CURRENT_RESULT_DIR: Path = BASE_RESULT_DIR / ACTIVE_DATASET
SPLITS = ("train", "eval", "test")
EVAL_SPLITS = ("eval",)

BUCKET_ORDER = ("many", "medium", "few")
BUCKET_THRESHOLDS = {
    "many": {"min": 100},
    "medium": {"min": 20, "max": 99},
    "few": {"max": 19},
}
BUCKET_ALIASES = {
    "many": "many",
    "many-shot": "many",
    "many_shot": "many",
    "medium": "medium",
    "medium-shot": "medium",
    "medium_shot": "medium",
    "few": "few",
    "few-shot": "few",
    "few_shot": "few",
}


def dataset_config(dataset_name: str | None = None) -> dict:
    name = dataset_name or ACTIVE_DATASET
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset={name!r}; choose one of {tuple(DATASETS)}")
    dataset = DATASETS[name]
    return {
        "dataset_name": name,
        "dataset_root": dataset["ROOT"],
        "bucket_order": BUCKET_ORDER,
        "bucket_thresholds": BUCKET_THRESHOLDS,
        "bucket_aliases": BUCKET_ALIASES,
    }


def current_dataset_runner_config() -> dict:
    return dataset_config(ACTIVE_DATASET)

# =========================================================
# BEATs
# =========================================================
BEATS = {

    "CHECKPOINT":
        CURRENT_DATASET["ROOT"] / f"BEATs_{ACTIVE_DATASET}.pt",

    "MODEL_DIR":
        BASE_DIR / "BEATs",

    "USE_AMP": False,

    "FP32_OUTPUT_DIR":
        CURRENT_RESULT_DIR / "fp32",

    "STATISTICS_DIR":
        CURRENT_RESULT_DIR / "statistics",
}


# =========================================================
# Logging
# =========================================================


# =========================================================
# Quantization Ablation
# =========================================================

# =========================================================
# Directory Creation
# =========================================================
ALL_DIRS = [
    CURRENT_RESULT_DIR,
    DATASETS["local_data"]["SPLIT_DIR"],
    DATASETS["PER_audio"]["SPLIT_DIR"],
    BEATS["FP32_OUTPUT_DIR"],
    BEATS["STATISTICS_DIR"],
]

for dir_path in ALL_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)

# =========================================================
# Deterministic Protocol
# =========================================================
def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


