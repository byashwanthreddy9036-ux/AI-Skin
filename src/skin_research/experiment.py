"""End-to-end experiment runner.

Pipeline (one model instance, one preprocessing pipeline, for every condition):

1. seed everything and snapshot the configuration;
2. load + validate the dataset and freeze the evaluation split (saved to CSV)
   *before* any variation is generated;
3. load the pretrained model once and check label compatibility;
4. stream source images: generate each condition's variant, assess its
   quality, optionally save it, and predict in fixed-size batches;
5. save predictions, manifest and run metadata;
6. evaluate, chart and report (when labels are compatible).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from skin_research.config import AppConfig, save_config_snapshot
from skin_research.data.loader import dataset_statistics, prepare_dataset
from skin_research.inference.model_loader import check_label_compatibility, load_model
from skin_research.inference.predictor import Predictor
from skin_research.preprocessing.variations import build_conditions, iter_variations, manifest_path, select_conditions
from skin_research.utils.logging import get_logger
from skin_research.utils.reproducibility import environment_info, set_global_seed

log = get_logger(__name__)

PREDICTION_FILE = "predictions.csv"
RUN_INFO_FILE = "run_info.json"


@dataclass
class ExperimentPaths:
    predictions_dir: Path
    reports_dir: Path
    charts_dir: Path

    @classmethod
    def for_config(cls, cfg: AppConfig) -> ExperimentPaths:
        name = cfg.experiment.name
        return cls(cfg.paths.predictions_dir / name, cfg.paths.reports_dir / name, cfg.paths.charts_dir / name)

    @property
    def predictions(self) -> Path:
        return self.predictions_dir / PREDICTION_FILE

    @property
    def run_info(self) -> Path:
        return self.predictions_dir / RUN_INFO_FILE


def run_experiment(cfg: AppConfig, condition_names: list[str] | None = None, force: bool = False,
                   evaluate: bool = True) -> dict[str, Any]:
    """Run the full experiment and return the run-info record."""
    t_start = time.perf_counter()
    paths = ExperimentPaths.for_config(cfg)
    if paths.run_info.is_file():
        try:
            previous = json.loads(paths.run_info.read_text())
            log.warning("Overwriting previous results of experiment '%s' (%s predictions). Use "
                        "--experiment-name to keep runs separate.", cfg.experiment.name,
                        previous.get("n_predictions"))
        except (OSError, json.JSONDecodeError):
            pass
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.seed)
    save_config_snapshot(cfg, paths.predictions_dir / "config_used.yaml")

    # 1-2. data and split (fixed before any transformation)
    log.info("Step 1/5: loading and validating dataset '%s'", cfg.dataset.name)
    prepared = prepare_dataset(cfg)
    split_csv = cfg.paths.processed_dir / f"{cfg.experiment.name}_split.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared.full[["image_id", "label", "lesion_id", "patient_id", "split_group", "split", "is_valid"]].to_csv(
        split_csv, index=False)
    table = prepared.table
    log.info("Evaluation split: %d images (%s-level grouping)", len(table), prepared.split_level)

    # 3. model
    log.info("Step 2/5: loading pretrained model %s", cfg.model.repo_id)
    adapter = load_model(cfg)
    compat = check_label_compatibility(adapter.labels, cfg.model.label_map, cfg.dataset.labels)
    if not compat.ok:
        log.error("Labels incompatible, metrics will NOT be computed: %s", "; ".join(compat.problems))
    predictor = Predictor(adapter, cfg.model.label_map, cfg.dataset.labels, cfg.model.batch_size)

    # 4. stream variations through the model
    conditions = select_conditions(build_conditions(cfg), condition_names)
    total = len(table) * len(conditions)
    log.info("Step 3/5: %d conditions × %d images = %d inferences (batch %d)",
             len(conditions), len(table), total, cfg.model.batch_size)
    frames = []
    for frame in predictor.predict_stream(iter_variations(cfg, table, conditions, force=force)):
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)

    # 5. persist
    log.info("Step 4/5: saving predictions (%d rows)", len(predictions))
    predictions.to_csv(paths.predictions, index=False)
    manifest_cols = [c for c in predictions.columns if not c.startswith("prob_")
                     and c not in {"predicted_label", "confidence", "margin", "entropy", "correct"}]
    mpath = manifest_path(cfg)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    predictions[manifest_cols].to_csv(mpath, index=False)

    run_info = {
        "experiment": cfg.experiment.name,
        "dataset": {
            "name": cfg.dataset.name,
            "metadata_csv": str(cfg.dataset.metadata_csv),
            "validation": prepared.validation.summary(),
            "evaluation_split_stats": dataset_statistics(table, cfg.dataset.labels),
            "split_level": prepared.split_level,
            "split_csv": str(split_csv),
        },
        "model": adapter.metadata(),
        "label_compatibility": {"ok": compat.ok, "problems": compat.problems},
        "conditions": [c.recipe() for c in conditions],
        "n_images": len(table),
        "n_conditions": len(conditions),
        "n_predictions": len(predictions),
        "inference_seconds": round(predictor.seconds, 1),
        "inference_images_per_second": round(predictor.throughput, 2),
        "seed": cfg.seed,
        "environment": environment_info(cfg.root),
    }

    if evaluate and compat.ok:
        log.info("Step 5/5: evaluating, charting and reporting")
        from skin_research.pipeline import evaluate_and_report

        run_info["outputs"] = evaluate_and_report(cfg, predictions, run_info)
    elif not compat.ok:
        run_info["outputs"] = {"note": "metrics skipped: labels incompatible"}

    run_info["total_seconds"] = round(time.perf_counter() - t_start, 1)
    paths.run_info.write_text(json.dumps(run_info, indent=2, default=str))
    log.info("Done in %.0f s → %s", run_info["total_seconds"], paths.predictions_dir)
    return run_info
