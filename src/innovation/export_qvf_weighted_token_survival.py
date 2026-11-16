from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import run_vanilla_kd as vanilla_kd
from src.innovation import export_QVF, export_qvf_token_survival
from src.innovation.run_qvf_weighted import DEFAULT_BETA, DEFAULT_LAYER_WEIGHTS, DEFAULT_SEED


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
        description="Weighted QVF token_survival wrapper. Defaults to beta=0.05 and L1:L9=0.25:0.75.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qvf-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--qvf-layer-weights", default=DEFAULT_LAYER_WEIGHTS)
    parser.add_argument("--output-root", default=None, help="Default: <dataset_root>/innovation/QVF_<weight_variant>.")
    parser.add_argument("--qvf-root", default=None, help="Default: <output-root>/seed_<seed>.")
    args, passthrough = parser.parse_known_args(argv)
    blocked = _owned_flag_present(
        passthrough,
        (
            "--betas",
            "--expected-method-variant",
            "--expected-protocol-version",
            "--expected-layer-weights",
            "--expected-qvf-weighted-variant",
        ),
    )
    if blocked:
        parser.error(f"{blocked} is owned by export_qvf_weighted_token_survival.py; use --qvf-beta/--qvf-layer-weights instead")
    return args, list(passthrough)


def _weighted_token_survival_argv(args: argparse.Namespace, passthrough: Sequence[str]) -> list[str]:
    layer_weights = vanilla_kd._parse_qvf_layer_weights(str(args.qvf_layer_weights))
    if not layer_weights:
        raise argparse.ArgumentTypeError("weighted QVF requires --qvf-layer-weights")
    variant = vanilla_kd.qvf_weighted_variant(layer_weights)
    dataset_root = export_QVF._dataset_root(args.dataset, None)
    output_root = Path(str(args.output_root)) if str(args.output_root or "").strip() else dataset_root / "innovation" / f"QVF_{variant}"
    qvf_root = Path(str(args.qvf_root)) if str(args.qvf_root or "").strip() else output_root / f"seed_{int(args.seed)}"
    method_variant = f"{vanilla_kd.QVF_WEIGHTED_METHOD_VARIANT_PREFIX}_{variant}"
    argv = list(passthrough)
    if args.dataset is not None:
        argv.extend(["--dataset", str(args.dataset)])
    argv.extend(
        [
            "--seed",
            str(int(args.seed)),
            "--qvf-root",
            str(qvf_root),
            "--betas",
            f"{float(args.qvf_beta):.10g}",
            "--expected-method-variant",
            method_variant,
            "--expected-protocol-version",
            vanilla_kd.QVF_WEIGHTED_PROTOCOL_VERSION,
            "--expected-layer-weights",
            str(args.qvf_layer_weights),
            "--expected-qvf-weighted-variant",
            variant,
        ]
    )
    return argv


def main(argv: Sequence[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    export_qvf_token_survival.main(_weighted_token_survival_argv(args, passthrough))


if __name__ == "__main__":
    main()
