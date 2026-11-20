"""Classification and reliability metrics.

All count-based metrics derive from one confusion-matrix routine
(:func:`metrics_from_confusion`) so that point estimates and bootstrap
resamples are computed identically. Macro averages use the classes present
in the ground truth of the evaluated sample: a class with no true examples
has undefined recall and is excluded rather than silently counted as 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix (rows = true class, columns = predicted class) from integer labels."""
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def per_class_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    """One-vs-rest statistics per class."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    total = cm.sum()
    fn, fp = support - tp, predicted - tp
    tn = total - tp - fn - fp
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    with np.errstate(invalid="ignore"):
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    return {"support": support, "precision": precision, "recall": recall,
            "specificity": _safe_div(tn, tn + fp), "f1": f1}


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """Aggregate metrics from a confusion matrix."""
    pc = per_class_from_confusion(cm)
    present = pc["support"] > 0
    total = cm.sum()
    # A class never predicted has precision 0 (not undefined) when it has true examples.
    precision = np.where(np.isnan(pc["precision"]) & present, 0.0, pc["precision"])
    return {
        "accuracy": float(np.trace(cm) / total) if total else float("nan"),
        "balanced_accuracy": float(np.nanmean(pc["recall"][present])) if present.any() else float("nan"),
        "macro_precision": float(np.mean(precision[present])) if present.any() else float("nan"),
        "macro_recall": float(np.nanmean(pc["recall"][present])) if present.any() else float("nan"),
        "macro_f1": float(np.mean(pc["f1"][present])) if present.any() else float("nan"),
        "macro_specificity": float(np.nanmean(pc["specificity"][present])) if present.any() else float("nan"),
    }


def calibration(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Expected calibration error, Brier score, NLL and reliability-diagram bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            acc, conf = correct[mask].mean(), confidence[mask].mean()
            ece += mask.mean() * abs(acc - conf)
            bins.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(mask.sum()),
                         "accuracy": float(acc), "confidence": float(conf)})
    onehot = np.eye(probs.shape[1])[y_true]
    return {
        "mean_confidence": float(confidence.mean()),
        "ece": float(ece),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1)).mean()),
        "reliability_bins": bins,
    }


def roc_auc(probs: np.ndarray, y_true: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """One-vs-rest ROC-AUC per class; macro average over classes where it is defined."""
    per_class: dict[str, float | None] = {}
    for k, label in enumerate(labels):
        positives = (y_true == k)
        if positives.all() or not positives.any():
            per_class[label] = None  # undefined without both positives and negatives
        else:
            per_class[label] = float(roc_auc_score(positives, probs[:, k]))
    defined = [v for v in per_class.values() if v is not None]
    return {"macro_roc_auc": float(np.mean(defined)) if defined else float("nan"), "roc_auc_per_class": per_class}


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, labels: list[str],
                         n_bins: int = 10) -> dict[str, Any]:
    """Complete metric set for one condition. ``y_true`` holds integer class indices."""
    y_pred = probs.argmax(axis=1)
    cm = confusion(y_true, y_pred, len(labels))
    pc = per_class_from_confusion(cm)
    result: dict[str, Any] = {"n": int(len(y_true)), **metrics_from_confusion(cm)}
    result.update(roc_auc(probs, y_true, labels))
    result.update(calibration(probs, y_true, n_bins))
    result["per_class"] = {
        label: {k: (None if np.isnan(v[i]) else float(v[i])) for k, v in pc.items()} for i, label in enumerate(labels)
    }
    result["confusion_matrix"] = cm.tolist()
    return result


def group_bootstrap_indices(groups: np.ndarray, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Cluster bootstrap: resample whole groups (lesions) with replacement.

    Images of the same lesion are correlated; resampling lesions rather than
    images gives honest (wider) confidence intervals.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    out = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(unique), len(unique))
        out.append(np.concatenate([members[g] for g in chosen]))
    return out


def bootstrap_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if not len(values):
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    return float(lo), float(hi)
