"""Website exporter tests.

Unit tests cover the experiment-status logic on synthetic files. The ``dataset`` tests verify the
built website data (website/data/site-data.js) against the real experiment CSVs value by value;
they are skipped, not passed, when the site has not been built.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest

from skin_research.config import load_config
from skin_research.experiment import ExperimentPaths, write_status
from skin_research.website.export import experiment_status, load_website_config, select_stories

from .conftest import LABELS


def _fake_outputs(cfg, n_images=14, n_conditions=2, n_rows=None, dataset_name=None):
    paths = ExperimentPaths.for_config(cfg)
    rep = cfg.paths.reports_dir / cfg.experiment.name
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    n_pred = n_images * n_conditions
    paths.run_info.write_text(json.dumps({"n_images": n_images, "n_conditions": n_conditions, "n_predictions": n_pred,
                                          "dataset": {"name": dataset_name or cfg.dataset.name}}))
    pd.DataFrame({"x": range(n_rows if n_rows is not None else n_pred)}).to_csv(paths.predictions, index=False)
    pd.DataFrame({"condition": [f"c{i}" for i in range(n_conditions)]}).to_csv(rep / "metrics_summary.csv", index=False)
    for name in ("metrics_vs_original.csv", "metrics_per_class.csv"):
        (rep / name).write_text("a\n1\n")
    (rep / "metrics.json").write_text("{}")
    return paths


def test_status_not_started_and_not_prepared(tmp_config):
    assert experiment_status(tmp_config)["state"] == "not_started"
    tmp_config.dataset.metadata_csv = tmp_config.root / "missing.csv"
    assert experiment_status(tmp_config)["state"] == "not_prepared"


def test_status_complete_only_when_every_check_passes(tmp_config):
    _fake_outputs(tmp_config)
    assert experiment_status(tmp_config)["state"] == "complete"


@pytest.mark.parametrize("kwargs, reason", [
    ({"n_rows": 5}, "rows"),                             # truncated predictions file
    ({"n_images": 7}, "of 14 dataset images"),          # trial run on a subset
    ({"dataset_name": "other"}, "different dataset"),    # outputs from another dataset
])
def test_status_incomplete_outputs_are_not_results(tmp_config, kwargs, reason):
    _fake_outputs(tmp_config, **kwargs)
    st = experiment_status(tmp_config)
    assert st["state"] == "incomplete" and reason in st["detail"]


def test_status_running_failed_and_interrupted(tmp_config):
    paths = _fake_outputs(tmp_config)
    write_status(paths.status, state="running", stage="inference", done=10, total=28, pid=os.getpid())
    st = experiment_status(tmp_config)
    assert st["state"] == "running" and st["done"] == 10 and st["total"] == 28
    write_status(paths.status, state="failed", error="RuntimeError: boom")
    assert experiment_status(tmp_config)["state"] == "failed"
    # a 'running' record whose process is gone and that is older than 15 minutes is reported as interrupted
    record = {"state": "running", "pid": 999999999, "updated_utc": "2020-01-01T00:00:00+00:00"}
    paths.status.write_text(json.dumps(record))
    assert experiment_status(tmp_config)["state"] == "interrupted"


def test_story_rules_use_real_predictions_only():
    rows = []
    for i, (label, orig_ok, blur_ok) in enumerate([("mel", True, False), ("nv", True, True), ("mel", False, False)]):
        for cond, ok, group in (("original", orig_ok, "original"), ("gaussian_blur__mild", blur_ok, "degradation")):
            probs = {f"prob_{l}": (0.9 if (l == label) == ok and l in (label, "bcc") else 0.0167) for l in LABELS}
            rows.append({"image_id": f"I{i}", "lesion_id": f"L{i}", "label": label, "condition": cond, "group": group,
                         "correct": ok, "confidence": 0.9, "predicted_label": label if ok else "bcc", **probs})
    stories, missing = select_stories(pd.DataFrame(rows), LABELS, max_lesions=8)
    by = {s["key"]: s for s in stories}
    assert by["blur"]["image_id"] == "I0"              # the only image correct → wrong under blur
    assert "fixed" in missing and "enhance_hurts" in missing   # no such conditions: reported, not invented


# ----------------------------------------------------------------------------- built site vs real CSVs
SITE_JS = load_website_config().output_dir / "data" / "site-data.js"
BUILT = SITE_JS.is_file()


def _site():
    text = SITE_JS.read_text(encoding="utf-8")
    return json.loads(re.search(r"window\.SITE_DATA = (\{.*\});\s*$", text, re.S).group(1))


@pytest.mark.dataset
@pytest.mark.skipif(not BUILT, reason="website not built; run `skin-research build-site`")
def test_every_exported_prediction_matches_predictions_csv():
    site = _site()
    labels = site["labels"]
    cfg = load_config()
    pred = pd.read_csv(ExperimentPaths.for_config(cfg).predictions).set_index(["image_id", "condition"])
    root = load_website_config().output_dir
    checked = 0
    for les in site["samples"]["lesions"]:
        assert set(les["conditions"]) == set(pred.loc[les["id"]].index)
        for cond, rec in les["conditions"].items():
            row = pred.loc[(les["id"], cond)]
            assert rec["pred"] == row["predicted_label"] and rec["correct"] == bool(row["correct"])
            assert rec["conf"] == pytest.approx(row["confidence"], abs=1e-4)
            assert rec["probs"] == pytest.approx([row[f"prob_{l}"] for l in labels], abs=1e-4)
            assert les["label"] == row["label"]
            assert (root / rec["img"]).is_file()
            checked += 1
    assert checked == len(site["samples"]["lesions"]) * len(site["catalog"])


@pytest.mark.dataset
@pytest.mark.skipif(not BUILT, reason="website not built; run `skin-research build-site`")
def test_exported_metrics_match_metric_tables_and_status():
    site = _site()
    wcfg = load_website_config()
    for exp in wcfg.experiments:
        cfg = load_config(wcfg.root / exp["config"])
        entry = site["experiments"][exp["key"]]
        if entry["status"]["state"] != "complete":
            assert "results" not in entry          # unfinished experiments never carry numbers
            continue
        rep = cfg.paths.reports_dir / cfg.experiment.name
        summary = pd.read_csv(rep / "metrics_summary.csv").set_index("condition")
        per_class = pd.read_csv(rep / "metrics_per_class.csv").set_index(["condition", "label"])
        for c in entry["results"]["conditions"]:
            s = summary.loc[c["name"]]
            assert c["bal"] == pytest.approx(s["balanced_accuracy"], abs=1e-4)
            assert c["bal_lo"] == pytest.approx(s["balanced_accuracy_ci_low"], abs=1e-4)
            assert c["acc"] == pytest.approx(s["accuracy"], abs=1e-4)
            assert c["ece"] == pytest.approx(s["ece"], abs=1e-4)
        for cond, by_label in entry["results"]["per_class"].items():
            for label, v in by_label.items():
                if v["recall"] is not None:
                    assert v["recall"] == pytest.approx(per_class.loc[(cond, label), "recall"], abs=1e-4)


@pytest.mark.dataset
@pytest.mark.skipif(not BUILT, reason="website not built; run `skin-research build-site`")
def test_curated_cases_satisfy_their_rules():
    site = _site()
    lesions = {l["id"]: l for l in site["samples"]["lesions"]}
    for s in site["samples"]["stories"]:
        c = lesions[s["image_id"]]["conditions"]
        if s["key"] in {"blur", "dark", "enhance_hurts", "noise", "confident_wrong"}:
            assert c["original"]["correct"] and not c[s["condition"]]["correct"], s
        elif s["key"] == "fixed":
            assert c[s["condition"]]["correct"] and not c["underexposure__moderate"]["correct"], s
        elif s["key"] in {"correct", "robust"}:
            assert c["original"]["correct"], s
    lesion_ids = [l["lesion_id"] for l in site["samples"]["lesions"]]
    assert len(lesion_ids) == len(set(lesion_ids))  # one photo per lesion in the gallery
