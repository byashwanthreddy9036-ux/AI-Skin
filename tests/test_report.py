from pathlib import Path

from skin_research.evaluation.comparison import evaluate_conditions
from skin_research.pipeline import evaluate_and_report
from skin_research.reporting.report import write_report

from .conftest import LABELS
from .test_evaluation import _fake_predictions


def test_report_and_charts_are_generated(tmp_config):
    predictions = _fake_predictions().assign(laplacian_var=100.0, brightness=0.6, rms_contrast=0.1, noise_sigma=1.0)
    run_info = {"n_images": 40, "n_conditions": 3, "n_predictions": 120,
                "model": {"repo_id": "test/model", "labels": ["MEL"], "parameters": 10},
                "dataset": {"split_level": "lesion", "evaluation_split_stats": {"class_counts": {"mel": 5}}},
                "conditions": []}
    outputs = evaluate_and_report(tmp_config, predictions, run_info)
    for chart in ("condition_overview", "severity", "enhancement", "classwise", "confidence", "reliability"):
        assert Path(outputs["charts"][chart]).stat().st_size > 1000, chart
    assert Path(outputs["metrics"]["json"]).is_file()
    report = Path(outputs["report"]).read_text()
    for section in ("## 1. Experiment configuration", "## 2. Dataset", "## 3. Model", "## 6. Classification",
                    "## 7. Automated observations", "## 8. Limitations", "## 9. Execution details"):
        assert section in report
    assert "Not for clinical use" in report and "gaussian_blur__mild" in report
    # image links in the report must resolve relative to the report file
    for line in report.splitlines():
        if line.startswith("!["):
            target = line[line.index("(") + 1:line.rindex(")")]
            assert (Path(outputs["report"]).parent / target).is_file(), target


def test_report_does_not_claim_improvement_without_evidence(tmp_config):
    predictions = _fake_predictions()
    result = evaluate_conditions(predictions, LABELS, n_resamples=100)
    text = write_report(tmp_config, predictions, result, {"conditions": []}, {"charts": {}}).read_text()
    assert "no transformation was shown to improve" in text
