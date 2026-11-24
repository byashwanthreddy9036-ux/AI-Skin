"""Real end-to-end smoke test: real ISIC images + verified checkpoint + real inference.

Skipped (reported as skipped, never as passed) if the dataset or weights are
not available locally.
"""

from pathlib import Path

import json

import pandas as pd
import pytest

from skin_research.config import config_from_dict, load_config
from skin_research.inference.model_loader import cached_model_files

BASE = load_config()
HAVE_DATA = BASE.dataset.metadata_csv.is_file() and BASE.dataset.image_root.is_dir()
HAVE_MODEL = cached_model_files(BASE)


@pytest.mark.model
@pytest.mark.dataset
@pytest.mark.skipif(not (HAVE_DATA and HAVE_MODEL), reason="needs ISIC data (prepare-data) and cached model weights")
def test_full_pipeline_on_real_images(tmp_path):
    raw = BASE.raw
    raw["paths"] = {k: str(tmp_path / k) if k not in ("models_dir", "raw_dir", "processed_dir") else v
                    for k, v in raw["paths"].items()}
    raw["paths"]["processed_dir"] = str(tmp_path / "processed")
    raw["dataset"]["max_images"] = 7
    raw["experiment"]["name"] = "smoke"
    raw["evaluation"]["bootstrap_resamples"] = 100
    cfg = config_from_dict(raw, root=BASE.root)

    from skin_research.experiment import run_experiment

    info = run_experiment(cfg, ["gaussian_blur__moderate", "underexposure__moderate", "enhance__sharpen",
                                "enhance__clahe", "underexposure__moderate+brightness_correction"])
    pred = pd.read_csv(cfg.paths.predictions_dir / "smoke" / "predictions.csv")
    assert info["label_compatibility"]["ok"]
    status = json.loads((cfg.paths.predictions_dir / "smoke" / "status.json").read_text())
    assert status["state"] == "complete" and status["done"] == status["total"] == len(pred)
    assert set(pred["group"]) == {"original", "degradation", "enhancement", "degrade_enhance"}
    assert len(pred) == info["n_images"] * 6
    probs = pred[[c for c in pred if c.startswith("prob_")]]
    assert ((probs.sum(axis=1) - 1).abs() < 1e-4).all()
    # originals must be the real dataset files, untouched
    orig = pred[pred.condition == "original"]
    assert all(Path(p).is_file() and str(BASE.dataset.image_root) in p for p in orig["output_path"])
    # a working model on held-out data must be far better than chance (1/7)
    assert orig["correct"].mean() > 0.5
    report = Path(info["outputs"]["report"]).read_text()
    assert "## 7. Automated observations" in report
    assert Path(info["outputs"]["metrics"]["json"]).is_file()

    # the explainer website builds from these real results
    import json
    import re

    from skin_research.reporting.site import build_site

    page = build_site(cfg)
    html = page.read_text()
    assert html.startswith("<!doctype html>") and "/*__SITE_DATA__*/" not in html
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert len(data["conditions"]) == 6 and data["examples"]
    first = data["examples"][0]["conditions"]["original"]
    assert (page.parent / first["img"]).is_file() and abs(sum(first["probs"]) - 1) < 1e-3
