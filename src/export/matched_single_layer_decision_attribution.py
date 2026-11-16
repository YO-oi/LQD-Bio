from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
import src.export.decision_harmful_token_ranking as ranking
import src.export.single_layer_causal_injection as single_layer
import src.export.three_factor_decomposition as three_factor
from src.utils.semantic_geometry_cache import file_identity, qparams_identity_manifest


EXPERIMENT_NAME = "matched_single_layer_decision_attribution"
PROTOCOL_VERSION = "matched_single_layer_decision_attribution.v1"
DEFAULT_OUTPUT_SUBDIR = EXPERIMENT_NAME
SINGLE_LAYER_SAMPLE_FILENAME = "single_layer_causal_injection.csv"
DEFAULT_SINGLE_LAYER_MATCH_TOLERANCE = 5e-6
DETAIL_FILENAME = "matched_single_layer_decision_attribution_detail.csv"
SUMMARY_FILENAME = "matched_single_layer_decision_attribution_summary.csv"
GROUP_SUMMARY_FILENAME = "matched_single_layer_decision_attribution_group_summary.csv"
MANIFEST_FILENAME = "manifest.json"


SUMMARY_COLUMNS = [
    *ranking.SUMMARY_COLUMNS,
    "single_layer_m_fp32",
    "single_layer_m_injected",
    "single_layer_margin_drop",
    "single_layer_reference_abs_error",
    "single_layer_m_fp32_abs_error",
    "single_layer_m_injected_abs_error",
    "single_layer_match_tolerance",
    "single_layer_reference_within_tolerance",
    "single_layer_reference_matched",
]


@dataclass(frozen=True)
class SingleLayerReferenceRow:
    sample_id: str
    target: str
    label_id: int
    fp32_pred: int
    injected_pred: int
    j_star: int
    m_fp32: float
    m_injected: float
    margin_drop: float
    classification_flip: int


def _resolve_oer_manifest_path(oer_oracle: Path) -> Path:
    candidates = (
        oer_oracle.with_suffix(oer_oracle.suffix + ".manifest.json"),
        oer_oracle.with_name("fp32_oer_oracle.manifest.json"),
        oer_oracle.with_suffix(".manifest.json"),
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = dict(single_layer._resolve_paths(args))
    output_dir = Path(args.output_dir) if args.output_dir is not None else paths["dataset_root"] / DEFAULT_OUTPUT_SUBDIR
    oer_oracle = Path(args.oer_oracle) if args.oer_oracle is not None else paths["dataset_root"] / ranking.DEFAULT_OER_ORACLE_RELATIVE
    three_factor_csv = (
        Path(args.three_factor_sample_csv)
        if args.three_factor_sample_csv is not None
        else paths["dataset_root"] / three_factor.DEFAULT_OUTPUT_SUBDIR / ranking.THREE_FACTOR_SAMPLE_FILENAME
    )
    single_layer_csv = (
        Path(args.single_layer_sample_csv)
        if args.single_layer_sample_csv is not None
        else paths["dataset_root"] / single_layer.DEFAULT_OUTPUT_SUBDIR / SINGLE_LAYER_SAMPLE_FILENAME
    )
    for label, path in (
        ("fixed OER oracle", oer_oracle),
        ("three-factor reference CSV", three_factor_csv),
        ("single-layer causal injection CSV", single_layer_csv),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    paths["output_dir"] = output_dir
    paths["oer_oracle"] = oer_oracle
    paths["oer_oracle_manifest"] = _resolve_oer_manifest_path(oer_oracle)
    paths["three_factor_sample_csv"] = three_factor_csv
    paths["single_layer_sample_csv"] = single_layer_csv
    return paths


def _validate_args(args: argparse.Namespace) -> None:
    ranking._validate_args(args)
    if float(args.single_layer_match_tolerance) <= 0.0:
        raise ValueError(
            f"single_layer_match_tolerance must be positive, got {args.single_layer_match_tolerance!r}"
        )
    if bool(getattr(args, "allow_full_w4a4", False)):
        raise ValueError(f"{EXPERIMENT_NAME} refuses full-W4A4 attribution")


def _load_single_layer_reference(path: Path) -> dict[tuple[str, str], SingleLayerReferenceRow]:
    rows: dict[tuple[str, str], SingleLayerReferenceRow] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "target",
            "label_id",
            "fp32_pred",
            "injected_pred",
            "j_star",
            "m_fp32",
            "m_injected",
            "margin_drop",
            "classification_flip",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"single-layer causal injection CSV missing required columns: {missing}")
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            target = str(row.get("target", "")).strip()
            if not sample_id or not target:
                continue
            key = (sample_id, target)
            if key in rows:
                raise RuntimeError(f"duplicate single-layer reference row for {sample_id} {target}")
            rows[key] = SingleLayerReferenceRow(
                sample_id=sample_id,
                target=target,
                label_id=int(float(row["label_id"])),
                fp32_pred=int(float(row["fp32_pred"])),
                injected_pred=int(float(row["injected_pred"])),
                j_star=int(float(row["j_star"])),
                m_fp32=float(row["m_fp32"]),
                m_injected=float(row["m_injected"]),
                margin_drop=float(row["margin_drop"]),
                classification_flip=int(float(row["classification_flip"])),
            )
    if not rows:
        raise ValueError(f"single-layer causal injection CSV contains no rows: {path}")
    return rows


def _target_keys(rows: Mapping[tuple[str, str], Any]) -> set[tuple[str, str]]:
    allowed = {spec.target for spec in single_layer.TARGET_SPECS}
    return {key for key in rows if key[1] in allowed}


def _attach_single_layer_match_columns(
    summary_rows: Sequence[Mapping[str, Any]],
    reference_rows: Mapping[tuple[str, str], SingleLayerReferenceRow],
    *,
    match_tolerance: float,
    strict_margin_match: bool,
    require_complete_reference: bool,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    produced_keys: set[tuple[str, str]] = set()
    for row in summary_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        target = str(row.get("target", "")).strip()
        key = (sample_id, target)
        if key in produced_keys:
            raise RuntimeError(f"duplicate matched single-layer output row for {sample_id} {target}")
        produced_keys.add(key)
        if key not in reference_rows:
            raise KeyError(f"{sample_id} {target}: missing single-layer causal injection reference row")
        ref = reference_rows[key]
        current_label = int(row["label_id"])
        current_fp32_pred = int(row["fp32_pred"])
        current_injected_pred = int(row["injected_pred"])
        current_j_star = int(row["j_star"])
        current_flip = int(row["classification_flip"])
        if current_label != int(ref.label_id):
            raise RuntimeError(f"{sample_id} {target}: label_id differs from single-layer reference")
        if current_fp32_pred != int(ref.fp32_pred):
            raise RuntimeError(f"{sample_id} {target}: fp32_pred differs from single-layer reference")
        if current_injected_pred != int(ref.injected_pred):
            raise RuntimeError(f"{sample_id} {target}: injected_pred differs from single-layer reference")
        if current_j_star != int(ref.j_star):
            raise RuntimeError(f"{sample_id} {target}: j_star differs from single-layer reference")
        if current_flip != int(ref.classification_flip):
            raise RuntimeError(f"{sample_id} {target}: classification_flip differs from single-layer reference")

        margin_error = abs(float(row["actual_margin_drop"]) - float(ref.margin_drop))
        fp32_error = abs(float(row["m_fp32"]) - float(ref.m_fp32))
        injected_error = abs(float(row["m_injected"]) - float(ref.m_injected))
        within_tolerance = (
            margin_error < float(match_tolerance)
            and fp32_error < float(match_tolerance)
            and injected_error < float(match_tolerance)
        )
        if strict_margin_match and margin_error >= float(match_tolerance):
            raise RuntimeError(
                f"{sample_id} {target}: actual_margin_drop differs from single-layer reference: "
                f"current={float(row['actual_margin_drop']):.8g}, ref={float(ref.margin_drop):.8g}, "
                f"abs_error={margin_error:.8g}"
            )
        if strict_margin_match and fp32_error >= float(match_tolerance):
            raise RuntimeError(
                f"{sample_id} {target}: m_fp32 differs from single-layer reference: "
                f"current={float(row['m_fp32']):.8g}, ref={float(ref.m_fp32):.8g}"
            )
        if strict_margin_match and injected_error >= float(match_tolerance):
            raise RuntimeError(
                f"{sample_id} {target}: m_injected differs from single-layer reference: "
                f"current={float(row['m_injected']):.8g}, ref={float(ref.m_injected):.8g}"
            )
        enriched.append(
            {
                **dict(row),
                "single_layer_m_fp32": float(ref.m_fp32),
                "single_layer_m_injected": float(ref.m_injected),
                "single_layer_margin_drop": float(ref.margin_drop),
                "single_layer_reference_abs_error": margin_error,
                "single_layer_m_fp32_abs_error": fp32_error,
                "single_layer_m_injected_abs_error": injected_error,
                "single_layer_match_tolerance": float(match_tolerance),
                "single_layer_reference_within_tolerance": bool(within_tolerance),
                "single_layer_reference_matched": True,
            }
        )

    if require_complete_reference:
        expected = _target_keys(reference_rows)
        if produced_keys != expected:
            missing = sorted(expected - produced_keys)[:8]
            extra = sorted(produced_keys - expected)[:8]
            raise RuntimeError(
                "matched output keys do not exactly equal single-layer causal injection keys: "
                f"missing={missing}, extra={extra}"
            )
    return enriched


def _build_manifest(
    *,
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    qparams_by_target: Mapping[single_layer.TargetSpec, Mapping[str, Any]],
    oer_oracle: ranking.OerOracleIndex,
    results: Sequence[ranking.TargetTokenRankingResult],
    shared_fp32_validation: Mapping[str, Any],
    oer_sample_order_validation: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
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
                "matched_single_layer_causal_injection": True,
                "representation_hook": {
                    "position": ranking.REPRESENTATION_HOOK_POSITION,
                    "semantic": "target Linear module output",
                },
                "oer_hook": {
                    "position": ranking.OER_HOOK_POSITION,
                    "semantic": "target FFN input activation",
                    "oracle_unit": "scalar",
                    "aggregation": "scalar OER indices aggregated to token-level oer_count before rank/overlap comparison",
                },
                "full_w4a4_forward_used": False,
                "quant_state": list(result.quant_state),
                "qparams_identity": qparams_identity_manifest(qparams, source_hint=spec.target),
                "single_layer_reference_validation": dict(result.single_layer_reference_validation),
                "checkpoint_clean_reloads": int(result.checkpoint_clean_reloads),
            }
        )

    match_errors = [
        float(row.get("single_layer_reference_abs_error", float("nan")))
        for row in summary_rows
        if math.isfinite(float(row.get("single_layer_reference_abs_error", float("nan"))))
    ]
    fp32_errors = [
        float(row.get("single_layer_m_fp32_abs_error", float("nan")))
        for row in summary_rows
        if math.isfinite(float(row.get("single_layer_m_fp32_abs_error", float("nan"))))
    ]
    injected_errors = [
        float(row.get("single_layer_m_injected_abs_error", float("nan")))
        for row in summary_rows
        if math.isfinite(float(row.get("single_layer_m_injected_abs_error", float("nan"))))
    ]
    outside_tolerance = [
        row for row in summary_rows
        if not bool(row.get("single_layer_reference_within_tolerance", False))
    ]
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
        "full_w4a4_attribution_system_used": False,
        "same_waveform": True,
        "same_center_crop": True,
        "same_padding_mask": True,
        "same_pooling": "mask_aware_mean",
        "same_classifier": True,
        "fixed_competitor": True,
        "competitor_definition": "j_star = argmax_{j != y} z_FP32[j]",
        "main_cohort": "FP32-correct eval samples matched to single_layer_causal_injection.csv",
        "groups": list(ranking.GROUPS),
        "tail_definition": "tail = medium + few",
        "flip_status_groups": list(ranking.FLIP_STATUSES),
        "token_topk": {
            "source": "project_config.TOKEN_TOPK_RATIO",
            "topk_ratio": float(ranking._fixed_token_topk_ratio()),
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
            "hook_position": ranking.OER_HOOK_POSITION,
            "activation_position": "input",
            "granularity": "scalar",
            "scalar_index_coordinate_system": "canonical BTC per sample: scalar_index = token_idx * feature_dim + channel_idx",
            "topk_ratios_seen": list(oer_oracle.topk_ratios),
            "shard_count": int(len(oer_oracle.shard_paths)),
            "shard_paths": list(oer_oracle.shard_paths),
            "rebuilt": False,
            "reselected": False,
        },
        "single_layer_causal_injection_reference": {
            "sample_csv": str(paths["single_layer_sample_csv"].resolve(strict=False)),
            "sample_csv_identity": file_identity(paths["single_layer_sample_csv"]),
            "required": True,
            "exact_key_set_match_required": True,
            "strict_margin_match": bool(args.strict_single_layer_margin_match),
            "match_tolerance": float(args.single_layer_match_tolerance),
            "fields_checked": [
                "label_id",
                "fp32_pred",
                "injected_pred",
                "j_star",
                "m_fp32",
                "m_injected",
                "margin_drop",
                "classification_flip",
            ],
            "max_abs_margin_drop_error": max(match_errors) if match_errors else float("nan"),
            "max_abs_m_fp32_error": max(fp32_errors) if fp32_errors else float("nan"),
            "max_abs_m_injected_error": max(injected_errors) if injected_errors else float("nan"),
            "rows_outside_match_tolerance": int(len(outside_tolerance)),
        },
        "three_factor_reference": {
            "sample_csv": str(paths["three_factor_sample_csv"].resolve(strict=False)),
            "sample_csv_identity": file_identity(paths["three_factor_sample_csv"]),
            "current_factor_algebra_checked": True,
            "first_order_csv_consistency_checked": False,
            "first_order_csv_difference_recorded": True,
            "actual_margin_drop_consistency_checked": False,
            "actual_margin_drop_difference_recorded": True,
        },
        "quantization": {
            "scope": "matched_isolated_single_ffn_module_per_target",
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
            "Hq_t": "isolated single-layer W4A4 target Linear module output",
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
            "sum_token_decision_projection_matches_current_factor_computation": True,
            "three_factor_csv_first_order_difference_recorded": True,
            "classification_fields_match_single_layer_causal_injection": True,
            "actual_margin_drop_computed_under_single_layer_protocol": True,
            "single_layer_margin_errors_recorded": True,
            "single_layer_margin_match_strict": bool(args.strict_single_layer_margin_match),
            "only_target_module_w4a4_per_experiment": True,
            "same_valid_token_mask_for_harmful_energy_distortion": True,
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
            "Matched single-layer attribution is descriptive. It is not a full-W4A4 attribution system, "
            "does not call high-energy tokens decision-critical, does not call harmful ranking causal token ranking, "
            "and does not propose a training loss from overlap."
        ),
    }


def _print_report(*, paths: Mapping[str, Path], summary_rows: Sequence[Mapping[str, Any]]) -> None:
    print("[Matched Single-Layer Decision Attribution]")
    print(f"checkpoint: {paths['checkpoint']}")
    print(f"single_layer_reference: {paths['single_layer_sample_csv']}")
    print(f"three_factor_reference: {paths['three_factor_sample_csv']}")
    print(f"oer_oracle: {paths['oer_oracle']}")
    print(f"output_dir: {paths['output_dir']}")
    print("test_touched: False")
    for spec in single_layer.TARGET_SPECS:
        rows = [row for row in summary_rows if row.get("target") == spec.target]
        flips = int(sum(int(row.get("classification_flip", 0)) for row in rows))
        errors = [
            float(row.get("single_layer_reference_abs_error", float("nan")))
            for row in rows
            if math.isfinite(float(row.get("single_layer_reference_abs_error", float("nan"))))
        ]
        mean_oer = np.mean(ranking._finite_values(rows, "harmful_oer_overlap")) if rows else float("nan")
        max_error = max(errors) if errors else float("nan")
        print(
            f"{spec.target}: n={len(rows)} flips={flips} "
            f"mean_harmful_oer_overlap={ranking._csv_value(float(mean_oer))} "
            f"max_single_layer_margin_error={ranking._csv_value(float(max_error))}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    ranking._assert_center_crop_protocol()
    paths = _resolve_paths(args)
    device = single_layer._device_from_args(args.device)
    metadata = single_layer._load_eval_metadata(paths["eval_csv"])
    qparams_by_target = single_layer._load_qparams_by_target(paths["qparams_root"])
    oer_oracle = ranking._load_oer_oracle_index(paths["oer_oracle"], paths["oer_oracle_manifest"])
    three_factor_reference = ranking._load_three_factor_reference(paths["three_factor_sample_csv"])
    single_layer_reference = _load_single_layer_reference(paths["single_layer_sample_csv"])
    loader = single_layer.build_loader(
        single_layer.dataset_paths(single_layer.DEFAULT_DATASET, None),
        split="eval",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    results: list[ranking.TargetTokenRankingResult] = []
    for spec in single_layer.TARGET_SPECS:
        result = ranking._run_target_token_ranking(
            spec=spec,
            checkpoint_path=paths["checkpoint"],
            qparams=qparams_by_target[spec],
            loader=loader,
            device=device,
            metadata=metadata,
            oer_oracle=oer_oracle,
            three_factor_reference=three_factor_reference,
            numerical_tolerance=float(args.numerical_tolerance),
            strict_three_factor_csv_first_order=False,
        )
        results.append(result)

    shared_fp32_validation = ranking._assert_shared_fp32_baseline(results)
    oer_sample_order_validation = ranking._assert_oer_sample_order(oer_oracle, results[0].sample_order)
    detail_rows = [row for result in results for row in result.detail_rows]
    raw_summary_rows = [row for result in results for row in result.summary_rows]
    summary_rows = _attach_single_layer_match_columns(
        raw_summary_rows,
        single_layer_reference,
        match_tolerance=float(args.single_layer_match_tolerance),
        strict_margin_match=bool(args.strict_single_layer_margin_match),
        require_complete_reference=True,
    )
    group_summary_rows = ranking._build_group_summary_rows(summary_rows)
    manifest = _build_manifest(
        paths=paths,
        args=args,
        device=device,
        qparams_by_target=qparams_by_target,
        oer_oracle=oer_oracle,
        results=results,
        shared_fp32_validation=shared_fp32_validation,
        oer_sample_order_validation=oer_sample_order_validation,
        summary_rows=summary_rows,
    )

    output_dir = paths["output_dir"]
    ranking._write_csv(output_dir / DETAIL_FILENAME, detail_rows, ranking.DETAIL_COLUMNS)
    ranking._write_csv(output_dir / SUMMARY_FILENAME, summary_rows, SUMMARY_COLUMNS)
    ranking._write_csv(output_dir / GROUP_SUMMARY_FILENAME, group_summary_rows, ranking.GROUP_SUMMARY_COLUMNS)
    ranking._write_json(output_dir / MANIFEST_FILENAME, manifest)
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
    parser.add_argument("--qparams-root", default=None, help="Directory containing frozen single-layer W4A4 qparams.")
    parser.add_argument("--oer-oracle", default=None, help="Fixed scalar FP32 OER oracle .pt. The script only reads it.")
    parser.add_argument("--three-factor-sample-csv", default=None, help="three_factor_sample.csv from src.export.three_factor_decomposition.")
    parser.add_argument("--single-layer-sample-csv", default=None, help="single_layer_causal_injection.csv reference to match exactly.")
    parser.add_argument("--output-dir", default=None, help="Directory for matched single-layer attribution CSVs.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=single_layer._default_batch_size())
    parser.add_argument("--num-workers", type=int, default=single_layer._default_num_workers())
    parser.add_argument("--seed", type=int, default=int(single_layer._cfg_get(project_config, "SPLIT_SEED", 42)))
    parser.add_argument("--numerical-tolerance", type=float, default=ranking.DEFAULT_NUMERICAL_TOLERANCE)
    parser.add_argument(
        "--single-layer-match-tolerance",
        type=float,
        default=DEFAULT_SINGLE_LAYER_MATCH_TOLERANCE,
        help=(
            "Tolerance used only when comparing recomputed margins with "
            "single_layer_causal_injection.csv. Classification fields still match exactly."
        ),
    )
    parser.add_argument(
        "--strict-single-layer-margin-match",
        action="store_true",
        help="Fail if any single-layer reference margin differs by --single-layer-match-tolerance.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
