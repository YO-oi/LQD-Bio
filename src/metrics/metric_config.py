from __future__ import annotations

from typing import Any


_FALLBACK_METRIC_DEFAULTS: dict[str, Any] = {
    "EPS": 1e-12,
    "SCALAR_OER_TOP_RATIO": 0.001,
    "TOKEN_SURVIVAL_MIN_NORM_RETENTION": 0.5,
    "MARGIN_MIN_ABS": 1e-6,
    "FISHER_NORMALIZE_BY_DOF": True,
    "GEOMETRY_VARIANCE_NORMALIZATION": "dof",
    "GEOMETRY_CENTER_SHIFT_MODE": "absolute",
    "SIGNED_INT4_SYMMETRIC_QMIN": -7,
    "SIGNED_INT4_SYMMETRIC_QMAX": 7,
}


def _load_project_metric_defaults() -> dict[str, Any]:
    defaults = dict(_FALLBACK_METRIC_DEFAULTS)
    try:
        import config as project_config
    except ModuleNotFoundError as exc:
        if exc.name != "config":
            raise
        return defaults

    quant_cfg = getattr(project_config, "QUANT", {})
    weight_cfg = quant_cfg.get("WEIGHT", {}) if isinstance(quant_cfg, dict) else {}
    metrics_cfg = getattr(project_config, "METRICS", {})
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = {}

    defaults["EPS"] = float(metrics_cfg.get("EPS", defaults["EPS"]))
    defaults["SCALAR_OER_TOP_RATIO"] = float(
        metrics_cfg.get(
            "SCALAR_OER_TOP_RATIO",
            getattr(project_config, "OER_SCALAR_TOP_RATIO", defaults["SCALAR_OER_TOP_RATIO"]),
        )
    )
    defaults["MARGIN_MIN_ABS"] = float(metrics_cfg.get("MARGIN_MIN_ABS", defaults["MARGIN_MIN_ABS"]))
    defaults["TOKEN_SURVIVAL_MIN_NORM_RETENTION"] = float(
        metrics_cfg.get("TOKEN_SURVIVAL_MIN_NORM_RETENTION", defaults["TOKEN_SURVIVAL_MIN_NORM_RETENTION"])
    )
    defaults["FISHER_NORMALIZE_BY_DOF"] = bool(
        metrics_cfg.get("FISHER_NORMALIZE_BY_DOF", defaults["FISHER_NORMALIZE_BY_DOF"])
    )
    defaults["GEOMETRY_VARIANCE_NORMALIZATION"] = str(
        metrics_cfg.get("GEOMETRY_VARIANCE_NORMALIZATION", defaults["GEOMETRY_VARIANCE_NORMALIZATION"])
    )
    defaults["GEOMETRY_CENTER_SHIFT_MODE"] = str(
        metrics_cfg.get(
            "GEOMETRY_CENTER_SHIFT_MODE",
            metrics_cfg.get("GEOMETRY_CENTER_DISTANCE_MODE", defaults["GEOMETRY_CENTER_SHIFT_MODE"]),
        )
    )
    defaults["SIGNED_INT4_SYMMETRIC_QMIN"] = int(
        metrics_cfg.get(
            "SIGNED_INT4_SYMMETRIC_QMIN",
            weight_cfg.get("QMIN", defaults["SIGNED_INT4_SYMMETRIC_QMIN"]),
        )
    )
    defaults["SIGNED_INT4_SYMMETRIC_QMAX"] = int(
        metrics_cfg.get(
            "SIGNED_INT4_SYMMETRIC_QMAX",
            weight_cfg.get("QMAX", defaults["SIGNED_INT4_SYMMETRIC_QMAX"]),
        )
    )
    return defaults


METRIC_DEFAULTS = _load_project_metric_defaults()
DEFAULT_EPS = float(METRIC_DEFAULTS["EPS"])
DEFAULT_SCALAR_OER_TOP_RATIO = float(METRIC_DEFAULTS["SCALAR_OER_TOP_RATIO"])
DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION = float(METRIC_DEFAULTS["TOKEN_SURVIVAL_MIN_NORM_RETENTION"])
DEFAULT_MARGIN_MIN_ABS = float(METRIC_DEFAULTS["MARGIN_MIN_ABS"])
DEFAULT_FISHER_NORMALIZE_BY_DOF = bool(METRIC_DEFAULTS["FISHER_NORMALIZE_BY_DOF"])
DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION = str(METRIC_DEFAULTS["GEOMETRY_VARIANCE_NORMALIZATION"])
DEFAULT_GEOMETRY_CENTER_SHIFT_MODE = str(METRIC_DEFAULTS["GEOMETRY_CENTER_SHIFT_MODE"])
DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN = int(METRIC_DEFAULTS["SIGNED_INT4_SYMMETRIC_QMIN"])
DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX = int(METRIC_DEFAULTS["SIGNED_INT4_SYMMETRIC_QMAX"])


__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_FISHER_NORMALIZE_BY_DOF",
    "DEFAULT_GEOMETRY_CENTER_SHIFT_MODE",
    "DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION",
    "DEFAULT_MARGIN_MIN_ABS",
    "DEFAULT_SCALAR_OER_TOP_RATIO",
    "DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN",
    "DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX",
    "METRIC_DEFAULTS",
]
