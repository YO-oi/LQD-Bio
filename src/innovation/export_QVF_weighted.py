from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import run_vanilla_kd as vanilla_kd
from src.innovation import export_QVF
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
        description="Weighted QVF formal export wrapper. Defaults to beta=0.05 and L1:L9=0.25:0.75.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qvf-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--qvf-layer-weights", default=DEFAULT_LAYER_WEIGHTS)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Default: <dataset_root>/innovation/QVF_<weight_variant>.",
    )
    args, passthrough = parser.parse_known_args(argv)
    blocked = _owned_flag_present(
        passthrough,
        (
            "--betas",
            "--formal-betas",
            "--expected-method-variant",
            "--expected-protocol-version",
            "--expected-layer-weights",
            "--expected-qvf-weighted-variant",
            "--vanilla-token-survival-dir",
        ),
    )
    if blocked:
        parser.error(f"{blocked} is owned by export_QVF_weighted.py; use --qvf-beta/--qvf-layer-weights instead")
    return args, list(passthrough)


def _weighted_export_argv(args: argparse.Namespace, passthrough: Sequence[str]) -> list[str]:
    layer_weights = vanilla_kd._parse_qvf_layer_weights(str(args.qvf_layer_weights))
    if not layer_weights:
        raise argparse.ArgumentTypeError("weighted QVF requires --qvf-layer-weights")
    variant = vanilla_kd.qvf_weighted_variant(layer_weights)
    dataset_root = export_QVF._dataset_root(args.dataset, args.dataset_root)
    output_root = Path(str(args.output_root)) if str(args.output_root or "").strip() else dataset_root / "innovation" / f"QVF_{variant}"
    method_variant = f"{vanilla_kd.QVF_WEIGHTED_METHOD_VARIANT_PREFIX}_{variant}"
    beta_text = f"{float(args.qvf_beta):.10g}"
    argv = list(passthrough)
    if args.dataset is not None:
        argv.extend(["--dataset", str(args.dataset)])
    if args.dataset_root is not None:
        argv.extend(["--dataset-root", str(args.dataset_root)])
    argv.extend(
        [
            "--seed",
            str(int(args.seed)),
            "--output-root",
            str(output_root),
            "--betas",
            beta_text,
            "--formal-betas",
            beta_text,
            "--expected-method-variant",
            method_variant,
            "--expected-protocol-version",
            vanilla_kd.QVF_WEIGHTED_PROTOCOL_VERSION,
            "--expected-layer-weights",
            str(args.qvf_layer_weights),
            "--expected-qvf-weighted-variant",
            variant,
            "--vanilla-token-survival-dir",
            str(dataset_root / "token_survival"),
        ]
    )
    return argv


def main(argv: Sequence[str] | None = None):
    args, passthrough = parse_args(argv)
    return export_QVF.main(_weighted_export_argv(args, passthrough))


if __name__ == "__main__":
    main()
