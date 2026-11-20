import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from skin_research.evaluation.comparison import evaluate_conditions, holm_adjust, mcnemar_exact
from skin_research.evaluation.metrics import (
    calibration,
    confusion,
    evaluate_predictions,
    group_bootstrap_indices,
    metrics_from_confusion,
)

from .conftest import LABELS


@pytest.mark.parametrize("seed", range(5))
def test_confusion_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 7, 200)
    y_pred = np.where(rng.random(200) < 0.6, y_true, rng.integers(0, 7, 200))
    m = metrics_from_confusion(confusion(y_true, y_pred, 7))
    present = np.unique(y_true)
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    assert m["macro_precision"] == pytest.approx(
        precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
    assert m["macro_recall"] == pytest.approx(recall_score(y_true, y_pred, labels=present, average="macro"))


def test_specificity_and_sensitivity_hand_example():
    # 2-class toy: true [0,0,0,1,1], pred [0,0,1,1,0]
    out = evaluate_predictions(np.array([0, 0, 0, 1, 1]),
                               np.array([[.9, .1], [.8, .2], [.4, .6], [.3, .7], [.6, .4]]), ["a", "b"])
    assert out["per_class"]["b"]["recall"] == pytest.approx(0.5)        # sensitivity for b: 1/2
    assert out["per_class"]["b"]["specificity"] == pytest.approx(2 / 3)  # 2 of 3 negatives correct
    assert out["confusion_matrix"] == [[2, 1], [1, 1]]
    assert 0 <= out["macro_roc_auc"] <= 1


def test_roc_auc_undefined_class_is_skipped():
    probs = np.array([[.8, .1, .1], [.2, .7, .1], [.6, .3, .1]])
    out = evaluate_predictions(np.array([0, 1, 0]), probs, ["a", "b", "c"])
    assert out["roc_auc_per_class"]["c"] is None
    assert out["per_class"]["c"]["recall"] is None   # no true examples → undefined, not 0


def test_calibration_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = calibration(np.eye(2)[y], y, n_bins=5)
    assert perfect["ece"] == pytest.approx(0) and perfect["brier"] == pytest.approx(0)
    wrong = calibration(np.eye(2)[1 - y] * 0.98 + 0.01, y, n_bins=5)
    assert wrong["ece"] > 0.9


def test_holm_and_mcnemar():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    a = np.array([True] * 10 + [False] * 10)
    b = np.array([False] * 10 + [False] * 10)
    a_only, b_only, p = mcnemar_exact(a, b)
    assert (a_only, b_only) == (10, 0) and p < 0.01
    assert mcnemar_exact(a, a)[2] == 1.0


def test_group_bootstrap_keeps_lesions_together():
    groups = np.array(["L1", "L1", "L2", "L3", "L3", "L3"])
    for idx in group_bootstrap_indices(groups, 20, seed=0):
        counts = pd.Series(groups[idx]).value_counts()
        assert all(counts.get(g, 0) % n == 0 for g, n in (("L1", 2), ("L3", 3)))


def _fake_predictions(n=40, seed=0):
    """Predictions table shaped like the runner's output (values chosen by the test)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 7, n)
    rows = []
    for cond, group, noise in (("original", "original", 0.1), ("gaussian_blur__mild", "degradation", 0.5),
                               ("gaussian_blur__mild+sharpen", "degrade_enhance", 0.3)):
        for i in range(n):
            p = np.full(7, noise / 6)
            pred = y[i] if rng.random() > noise else rng.integers(0, 7)
            p[pred] = 1 - noise if pred == y[i] or noise < 1 else p[pred]
            p = p / p.sum()
            rows.append({"image_id": f"I{i}", "lesion_id": f"L{i // 2}", "label": LABELS[y[i]], "condition": cond,
                         "group": group, "degradation": "gaussian_blur" if group != "original" else None,
                         "severity": "mild" if group != "original" else None,
                         "enhancement": "sharpen" if group == "degrade_enhance" else None,
                         "confidence": p.max(), "correct": LABELS[p.argmax()] == LABELS[y[i]],
                         "label_usefulness": "ok", **{f"prob_{lbl}": p[k] for k, lbl in enumerate(LABELS)}})
    return pd.DataFrame(rows)


def test_evaluate_conditions_end_to_end():
    res = evaluate_conditions(_fake_predictions(), LABELS, n_resamples=200, seed=0)
    assert list(res.summary["condition"]) == ["original", "gaussian_blur__mild", "gaussian_blur__mild+sharpen"]
    orig = res.summary.iloc[0]
    assert orig["balanced_accuracy_ci_low"] <= orig["balanced_accuracy"] <= orig["balanced_accuracy_ci_high"]
    row = res.vs_original.set_index("condition").loc["gaussian_blur__mild"]
    assert row["delta_balanced_accuracy"] == pytest.approx(
        res.summary.iloc[1]["balanced_accuracy"] - orig["balanced_accuracy"])
    assert 0 <= row["flip_rate"] <= 1 and "mcnemar_p_holm" in res.vs_original
    assert res.restoration.iloc[0]["degraded_condition"] == "gaussian_blur__mild"
    assert res.details["bootstrap"]["unit"] == "lesion"


def test_evaluation_refuses_unknown_labels_and_missing_baseline():
    preds = _fake_predictions()
    with pytest.raises(ValueError, match="not in the evaluated label set"):
        evaluate_conditions(preds.assign(label="psoriasis"), LABELS, n_resamples=10)
    with pytest.raises(ValueError, match="baseline"):
        evaluate_conditions(preds[preds.condition != "original"], LABELS, n_resamples=10)
