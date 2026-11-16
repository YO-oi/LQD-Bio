from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from src.baseline.manifest import canonical_json_hash, file_sha256, git_commit, path_str


QPARAM_COLUMNS = (
    "layer_name",
    "layer_index",
    "module_path",
    "linear_role",
    "in_features",
    "out_features",
    "weight_bit",
    "activation_bit",
    "weight_granularity",
    "activation_granularity",
    "qmin",
    "qmax",
    "symmetric",
    "observer_type",
    "calibration_csv_hash",
    "weight_qmin",
    "weight_qmax",
    "weight_symmetric",
    "weight_scale",
    "weight_zero_point",
    "activation_qmin",
    "activation_qmax",
    "activation_symmetric",
    "activation_scale",
    "activation_zero_point",
)


def tensor_to_json(value: Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    return json.dumps(tensor.tolist(), ensure_ascii=False, separators=(",", ":"))


def tensor_from_json(value: str) -> Tensor:
    return torch.as_tensor(json.loads(value), dtype=torch.float32)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().float().contiguous()
        return tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    return value


def qparams_to_jsonable(qparams: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(qparams, Mapping) or not qparams:
        raise ValueError("qparams must be a non-empty mapping")
    return _jsonable(qparams)


def qparams_hash(qparams: Mapping[str, Any]) -> str:
    return canonical_json_hash(qparams_to_jsonable(qparams))


def _layer_index_from_module_path(module_path: str) -> int:
    parts = str(module_path).split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "layers" and parts[index + 1].isdigit():
            return int(parts[index + 1])
    raise ValueError(f"cannot infer layer index from module path: {module_path}")


def _linear_role_from_module_path(module_path: str) -> str:
    role = str(module_path).split(".")[-1]
    if role not in {"fc1", "fc2"}:
        raise ValueError(f"Baseline INT4 only allows FFN fc1/fc2, got module path: {module_path}")
    return role


def module_qparams(module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    weight = module.linear.weight.detach()
    weight_scale, weight_zero_point = module._weight_qparams_for(weight)
    cfg = module.config
    weight_qparams = {
        "scale": weight_scale.detach().cpu().float().clone(),
        "zero_point": weight_zero_point.detach().cpu().float().clone(),
        "qmin": int(cfg.weight_qmin),
        "qmax": int(cfg.weight_qmax),
        "symmetric": bool(cfg.weight_symmetric),
        "granularity": "per_channel",
        "observer_type": "weight",
        "channel_axis": 0,
    }
    activation_qparams = {
        "scale": module.activation_scale.detach().cpu().float().clone(),
        "zero_point": module.activation_zero_point.detach().cpu().float().clone(),
        "qmin": int(cfg.activation_qmin),
        "qmax": int(cfg.activation_qmax),
        "symmetric": bool(cfg.activation_symmetric),
        "granularity": "per_tensor",
        "observer_type": "activation",
    }
    return weight_qparams, activation_qparams


def quantized_layer_record(module_path: str, module: Any, *, calibration_csv_hash: str | None) -> dict[str, Any]:
    cfg = module.config
    layer_index = _layer_index_from_module_path(module_path)
    role = _linear_role_from_module_path(module_path)
    return {
        "layer_name": f"encoder.layers.{layer_index}",
        "layer_index": int(layer_index),
        "module_path": str(module_path),
        "linear_role": role,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "weight_bit": int(cfg.weight_bits),
        "activation_bit": int(cfg.activation_bits),
        "weight_granularity": "per_channel",
        "activation_granularity": "per_tensor",
        "qmin": int(cfg.weight_qmin),
        "qmax": int(cfg.weight_qmax),
        "symmetric": bool(cfg.weight_symmetric and cfg.activation_symmetric),
        "observer_type": "two_pass_histogram_activation_plus_weight_minmax",
        "calibration_csv_hash": calibration_csv_hash,
    }


def collect_quantized_layers(controller: Any, *, calibration_csv_hash: str | None) -> list[dict[str, Any]]:
    rows = [
        quantized_layer_record(module_path, module, calibration_csv_hash=calibration_csv_hash)
        for module_path, module in sorted(controller.modules.items())
    ]
    if not rows:
        raise RuntimeError("Baseline INT4 found no quantized FFN fc1/fc2 modules")
    bad_roles = [row for row in rows if row["linear_role"] not in {"fc1", "fc2"}]
    if bad_roles:
        raise RuntimeError(f"Baseline INT4 target scope violation: {bad_roles}")
    return rows


def collect_qparams_from_controller(controller: Any) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for module_path, module in sorted(controller.modules.items()):
        weight_qparams, activation_qparams = module_qparams(module)
        result[str(module_path)] = {"weight": weight_qparams, "activation": activation_qparams}
    if not result:
        raise RuntimeError("Baseline INT4 found no qparams to serialize")
    return result


def write_quantized_layers_json(path: Path, layers: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([dict(row) for row in layers], handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_qparams_csv(path: Path, controller: Any, *, calibration_csv_hash: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QPARAM_COLUMNS))
        writer.writeheader()
        for module_path, module in sorted(controller.modules.items()):
            layer = quantized_layer_record(module_path, module, calibration_csv_hash=calibration_csv_hash)
            weight_qparams, activation_qparams = module_qparams(module)
            writer.writerow(
                {
                    **layer,
                    "weight_qmin": int(weight_qparams["qmin"]),
                    "weight_qmax": int(weight_qparams["qmax"]),
                    "weight_symmetric": bool(weight_qparams["symmetric"]),
                    "weight_scale": tensor_to_json(weight_qparams["scale"]),
                    "weight_zero_point": tensor_to_json(weight_qparams["zero_point"]),
                    "activation_qmin": int(activation_qparams["qmin"]),
                    "activation_qmax": int(activation_qparams["qmax"]),
                    "activation_symmetric": bool(activation_qparams["symmetric"]),
                    "activation_scale": tensor_to_json(activation_qparams["scale"]),
                    "activation_zero_point": tensor_to_json(activation_qparams["zero_point"]),
                }
            )


def load_qparams_csv(path: str | Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    target = Path(path)
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            module_path = str(row["module_path"])
            result[module_path] = {
                "weight": {
                    "scale": tensor_from_json(str(row["weight_scale"])),
                    "zero_point": tensor_from_json(str(row["weight_zero_point"])),
                    "qmin": int(row["weight_qmin"]),
                    "qmax": int(row["weight_qmax"]),
                    "symmetric": str(row["weight_symmetric"]).lower() in {"1", "true", "yes"},
                    "granularity": "per_channel",
                    "observer_type": "weight",
                    "channel_axis": 0,
                },
                "activation": {
                    "scale": tensor_from_json(str(row["activation_scale"])),
                    "zero_point": tensor_from_json(str(row["activation_zero_point"])),
                    "qmin": int(row["activation_qmin"]),
                    "qmax": int(row["activation_qmax"]),
                    "symmetric": str(row["activation_symmetric"]).lower() in {"1", "true", "yes"},
                    "granularity": "per_tensor",
                    "observer_type": "activation",
                },
            }
    if not result:
        raise RuntimeError(f"qparams CSV produced no module qparams: {target}")
    return result


def qparams_csv_metadata(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    calibration_hashes = sorted({str(row.get("calibration_csv_hash", "")) for row in rows if str(row.get("calibration_csv_hash", "")).strip()})
    return {
        "num_rows": int(len(rows)),
        "calibration_hash": calibration_hashes[0] if len(calibration_hashes) == 1 else None,
        "calibration_hashes": calibration_hashes,
        "module_paths": sorted(str(row.get("module_path", "")) for row in rows),
    }


def checkpoint_identity_payload(
    *,
    model_kind: str,
    cfg: Mapping[str, Any],
    label_mapping: Mapping[str, int],
    label_mapping_hash: str | None,
    source_beats_checkpoint: str | Path | None,
    teacher_checkpoint: str | Path | None,
    quantized_layers: Sequence[Mapping[str, Any]],
    qparams: Mapping[str, Any] | None,
    qparams_csv: str | Path | None,
    calibration_csv: str | Path | None,
    calibration_hash: str | None,
    pooling: str,
    seed: int,
) -> dict[str, Any]:
    layers = [dict(item) for item in quantized_layers]
    quant_cfg = getattr(__import__("config"), "QUANT", {})
    weight_cfg = dict(quant_cfg.get("WEIGHT", {})) if isinstance(quant_cfg, Mapping) else {}
    activation_cfg = dict(quant_cfg.get("ACTIVATION", {})) if isinstance(quant_cfg, Mapping) else {}
    qparams_hash_value = qparams_hash(qparams) if qparams is not None else None
    source_beats_checkpoint_sha256 = file_sha256(source_beats_checkpoint)
    teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint)
    qparams_csv_sha256 = file_sha256(qparams_csv)
    audio_input_py = Path(__file__).resolve().parents[2] / "src" / "data" / "audio_input.py"
    group_utils_py = Path(__file__).resolve().parents[2] / "src" / "utils" / "group_utils.py"
    audio_input_py_sha256 = file_sha256(audio_input_py)
    group_utils_py_sha256 = file_sha256(group_utils_py)
    return {
        "model_kind": str(model_kind),
        "cfg_hash": canonical_json_hash(dict(cfg)),
        "label_mapping_hash": label_mapping_hash,
        "label_mapping_dense_hash": canonical_json_hash({str(k): int(v) for k, v in sorted(label_mapping.items())}),
        "source_beats_checkpoint": path_str(source_beats_checkpoint),
        "source_beats_checkpoint_sha256": source_beats_checkpoint_sha256,
        "source_beats_checkpoint_hash": source_beats_checkpoint_sha256,
        "teacher_checkpoint": path_str(teacher_checkpoint),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_checkpoint_hash": teacher_checkpoint_sha256,
        "quantized_layers_hash": canonical_json_hash(layers),
        "bitwidth": 4 if layers else "fp32",
        "qmin": int(weight_cfg.get("QMIN", -7)),
        "qmax": int(weight_cfg.get("QMAX", 7)),
        "symmetric": bool(weight_cfg.get("SYMMETRIC", True) and activation_cfg.get("SYMMETRIC", True)),
        "weight_granularity": str(weight_cfg.get("GRANULARITY", "per_channel")),
        "activation_granularity": str(activation_cfg.get("GRANULARITY", "per_tensor")),
        "qparams_hash": qparams_hash_value,
        "qparams_csv": path_str(qparams_csv),
        "qparams_csv_sha256": qparams_csv_sha256,
        "qparams_csv_hash": qparams_csv_sha256,
        "calibration_csv": path_str(calibration_csv),
        "calibration_hash": calibration_hash,
        "audio_input_py_sha256": audio_input_py_sha256,
        "audio_input_hash": audio_input_py_sha256,
        "group_utils_py_sha256": group_utils_py_sha256,
        "group_utils_hash": group_utils_py_sha256,
        "pooling": str(pooling),
        "seed": int(seed),
        "git_commit": git_commit(),
    }


__all__ = [
    "QPARAM_COLUMNS",
    "checkpoint_identity_payload",
    "collect_qparams_from_controller",
    "collect_quantized_layers",
    "load_qparams_csv",
    "qparams_csv_metadata",
    "qparams_hash",
    "qparams_to_jsonable",
    "tensor_from_json",
    "tensor_to_json",
    "write_qparams_csv",
    "write_quantized_layers_json",
]
