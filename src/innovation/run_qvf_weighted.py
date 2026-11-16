from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import run_vanilla_kd as vanilla_kd


DEFAULT_LAYER_WEIGHTS = "1:0.25,9:0.75"
DEFAULT_BETA = 0.05
DEFAULT_SEED = 2026


def _owned_flag_present(argv: Sequence[str], names: Sequence[str]) -> str | None:
    names = tuple(str(item) for item in names)
    for item in argv:
        text = str(item)
        if text in names:
            return text
        if any(text.startswith(f"{name}=") for name in names):
            return text
    return None


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Weighted QVF Stage1 trainer. Defaults to beta=0.05 and L1:L9=0.25:0.75.",
        add_help=True,
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qvf-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--qvf-layer-weights", default=DEFAULT_LAYER_WEIGHTS)
    parser.add_argument(
        "--qvf-output-root",
        default=None,
        help="Default: <dataset_root>/innovation/QVF_<weight_variant>.",
    )
    args, passthrough = parser.parse_known_args(argv)
    blocked = _owned_flag_present(
        passthrough,
        (
            "--baseline-root",
            "--qvf-method-variant",
            "--qvf-protocol-version",
            "--qvf-weighted-variant",
        ),
    )
    if blocked:
        parser.error(f"{blocked} is owned by run_qvf_weighted.py; use --qvf-layer-weights/--qvf-output-root instead")
    return args, list(passthrough)


def _weighted_baseline_argv(args: argparse.Namespace, passthrough: Sequence[str]) -> list[str]:
    layer_weights = vanilla_kd._parse_qvf_layer_weights(str(args.qvf_layer_weights))
    if not layer_weights:
        raise argparse.ArgumentTypeError("weighted QVF requires --qvf-layer-weights")
    variant = vanilla_kd.qvf_weighted_variant(layer_weights)
    output_root = (
        Path(str(args.qvf_output_root))
        if str(args.qvf_output_root or "").strip()
        else vanilla_kd._dataset_root(args.dataset) / "innovation" / f"QVF_{variant}"
    )
    method_variant = f"{vanilla_kd.QVF_WEIGHTED_METHOD_VARIANT_PREFIX}_{variant}"
    argv = list(passthrough)
    if args.dataset is not None:
        argv.extend(["--dataset", str(args.dataset)])
    argv.extend(
        [
            "--seed",
            str(int(args.seed)),
            "--use-qvf-token-fidelity",
            "--qvf-beta",
            f"{float(args.qvf_beta):.10g}",
            "--qvf-layer-weights",
            str(args.qvf_layer_weights),
            "--qvf-output-root",
            str(output_root),
            "--qvf-method-variant",
            method_variant,
            "--qvf-protocol-version",
            vanilla_kd.QVF_WEIGHTED_PROTOCOL_VERSION,
            "--qvf-weighted-variant",
            variant,
        ]
    )
    return argv


def main(argv: Sequence[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    baseline_args = vanilla_kd.parse_args(_weighted_baseline_argv(args, passthrough))
    vanilla_kd.run_vanilla_kd(baseline_args)


if __name__ == "__main__":
    main()
