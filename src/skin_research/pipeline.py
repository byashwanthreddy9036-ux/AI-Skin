"""Post-inference stages shared by `experiment`, `evaluate` and `report`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skin_research.config import AppConfig
from skin_research.evaluation.comparison import EvaluationResult, evaluate_conditions
from skin_research.reporting import charts
from skin_research.utils.logging import get_logger

log = get_logger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def results_dir(cfg: AppConfig) -> Path:
    return cfg.paths.reports_dir / cfg.experiment.name


def charts_dir(cfg: AppConfig) -> Path:
    return cfg.paths.charts_dir / cfg.experiment.name


def save_evaluation(cfg: AppConfig, result: EvaluationResult) -> dict[str, str]:
    """Write every evaluation table as CSV plus one combined JSON file."""
    out = results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("summary", "per_class", "vs_original", "restoration", "severity", "quality_strata"):
        table: pd.DataFrame = getattr(result, name)
        path = out / f"metrics_{name}.csv"
        table.to_csv(path, index=name in {"severity", "quality_strata"})
        files[name] = str(path)
    payload = {
        "summary": result.summary.to_dict(orient="records"),
        "vs_original": result.vs_original.to_dict(orient="records"),
        "restoration": result.restoration.to_dict(orient="records"),
        **result.details,
    }
    json_path = out / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    files["json"] = str(json_path)
    return files


def pick_example_image(predictions: pd.DataFrame) -> str | None:
    """A lesion whose original prediction was correct and that flips under some degradation."""
    orig = predictions[predictions["condition"] == "original"].set_index("image_id")
    flips = predictions[(predictions["group"] == "degradation") & ~predictions["correct"]]
    candidates = [i for i in flips["image_id"].unique() if orig.loc[i, "correct"]]
    if not candidates:
        return orig.index[0] if len(orig) else None
    # prefer a non-majority class to make the example informative
    minority = [i for i in candidates if orig.loc[i, "label"] != orig["label"].mode().iloc[0]]
    return sorted(minority or candidates)[0]


def make_charts(cfg: AppConfig, predictions: pd.DataFrame, result: EvaluationResult) -> dict[str, str]:
    out = charts_dir(cfg)
    metric = cfg.evaluation.primary_metric
    labels = cfg.dataset.labels
    summary = result.summary
    files: dict[str, str] = {}

    def add(key: str, path: Path | None) -> None:
        if path is not None:
            files[key] = str(path)

    add("condition_overview", charts.condition_overview(summary, metric, out / "01_condition_overview.png"))
    if (summary["group"] == "degradation").any():
        add("severity", charts.severity_small_multiples(summary, metric, out / "02_severity.png"))
    if (summary["group"].isin(["enhancement", "degrade_enhance"])).any():
        add("enhancement", charts.enhancement_comparison(summary, result.restoration, metric,
                                                         out / "03_enhancement.png"))
    conditions = summary["condition"].tolist()
    support = predictions[predictions["condition"] == "original"]["label"].value_counts().to_dict()
    add("classwise", charts.classwise_heatmap(result.per_class, "recall", conditions, labels, support,
                                              out / "04_classwise_recall.png"))
    focus = ["original"] + [c for c in conditions if c.endswith("__moderate") or c.startswith("enhance__")]
    add("confidence", charts.confidence_comparison(predictions, focus, out / "05_confidence.png"))
    worst = summary[summary["condition"] != "original"].sort_values(metric).iloc[0]["condition"] \
        if len(summary) > 1 else None
    best_enh = summary[summary["group"].isin(["enhancement", "degrade_enhance"])].sort_values(metric)
    rel = ["original", *([worst] if worst else []), *(best_enh["condition"].tail(1).tolist())]
    add("reliability", charts.reliability_diagram(result.details["per_condition"], list(dict.fromkeys(rel)),
                                                  out / "06_reliability.png"))
    add("confusion_original", charts.confusion_matrix_plot(
        result.details["per_condition"]["original"]["confusion_matrix"], labels, "Confusion matrix — original",
        out / "07_confusion_original.png"))
    example = pick_example_image(predictions)
    if example and "output_path" in predictions and predictions["output_path"].notna().any():
        grid_conditions = ["original"] + [c for c in conditions if c.endswith("__severe")][:4] + \
                          [c for c in conditions if c.startswith("enhance__")][:3]
        add("example", charts.example_grid(predictions, example, grid_conditions, out / "08_example.png"))
    orig = predictions[predictions["condition"] == "original"]
    if "laplacian_var" in orig:
        add("quality", charts.quality_distribution(orig, out / "00_quality_distribution.png"))
    return files


def evaluate_and_report(cfg: AppConfig, predictions: pd.DataFrame, run_info: dict[str, Any]) -> dict[str, Any]:
    """Evaluate predictions, save tables and charts, and write the report."""
    ev = cfg.evaluation
    result = evaluate_conditions(predictions, cfg.dataset.labels, n_resamples=ev.bootstrap_resamples,
                                 level=ev.confidence_level, n_bins=ev.calibration_bins, seed=cfg.seed,
                                 primary=ev.primary_metric)
    outputs: dict[str, Any] = {"metrics": save_evaluation(cfg, result)}
    outputs["charts"] = make_charts(cfg, predictions, result)
    try:
        from skin_research.reporting.report import write_report
    except ImportError:  # report module added in Phase 9
        return outputs
    outputs["report"] = str(write_report(cfg, predictions, result, run_info, outputs))
    return outputs
