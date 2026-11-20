"""Condition-level evaluation and paired statistical comparisons.

Every condition contains the *same* source images, so comparisons with the
original are paired: the bootstrap resamples the same lesions for both
conditions, and McNemar's exact test uses per-image correctness pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from skin_research.evaluation.metrics import (
    bootstrap_ci,
    confusion,
    evaluate_predictions,
    group_bootstrap_indices,
    metrics_from_confusion,
)
from skin_research.utils.logging import get_logger

log = get_logger(__name__)

BOOTSTRAP_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
BASELINE = "original"


@dataclass
class EvaluationResult:
    summary: pd.DataFrame                      # one row per condition
    per_class: pd.DataFrame                    # condition × class rows
    vs_original: pd.DataFrame                  # paired comparison with baseline
    restoration: pd.DataFrame                  # degrade+enhance vs degraded and original
    severity: pd.DataFrame                     # degradation × severity pivot of primary metric
    quality_strata: pd.DataFrame               # accuracy for ok vs questionable-quality images
    details: dict[str, Any] = field(default_factory=dict)  # full metric dicts (JSON)


def _arrays(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {lbl: i for i, lbl in enumerate(labels)}
    y_true = df["label"].map(index).to_numpy()
    probs = df[[f"prob_{lbl}" for lbl in labels]].to_numpy(dtype=float)
    return y_true, probs


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjustment for multiple comparisons."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on paired correctness. Returns (a_only, b_only, p-value)."""
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p


def evaluate_conditions(predictions: pd.DataFrame, labels: list[str], *, n_resamples: int = 1000,
                        level: float = 0.95, n_bins: int = 10, seed: int = 42,
                        primary: str = "balanced_accuracy") -> EvaluationResult:
    """Compute all metrics, CIs and comparisons for a predictions table."""
    if BASELINE not in set(predictions["condition"]):
        raise ValueError("Predictions must include the 'original' condition as the baseline.")
    unknown = set(predictions["label"]) - set(labels)
    if unknown:
        raise ValueError(f"Ground-truth labels {sorted(unknown)} are not in the evaluated label set.")

    conditions = list(dict.fromkeys(predictions["condition"]))
    by_cond = {c: predictions[predictions["condition"] == c].set_index("image_id") for c in conditions}
    base = by_cond[BASELINE]
    image_ids = base.index.to_numpy()
    for c, df in by_cond.items():
        if set(df.index) != set(image_ids):
            raise ValueError(f"Condition '{c}' does not cover the same images as 'original'.")
        by_cond[c] = df.loc[image_ids]

    groups = base["lesion_id"].fillna(pd.Series(base.index, index=base.index)).astype(str).to_numpy() \
        if "lesion_id" in base else image_ids.astype(str)
    resamples = group_bootstrap_indices(groups, n_resamples, seed)
    k = len(labels)

    y_true, _ = _arrays(base, labels)
    boot: dict[str, dict[str, np.ndarray]] = {}
    preds: dict[str, np.ndarray] = {}
    summary_rows, per_class_rows, details = [], [], {}
    for c in conditions:
        df = by_cond[c]
        _, probs = _arrays(df, labels)
        y_pred = probs.argmax(axis=1)
        preds[c] = y_pred
        m = evaluate_predictions(y_true, probs, labels, n_bins)
        details[c] = m
        boot[c] = {name: np.empty(n_resamples) for name in BOOTSTRAP_METRICS}
        for r, idx in enumerate(resamples):
            bm = metrics_from_confusion(confusion(y_true[idx], y_pred[idx], k))
            for name in BOOTSTRAP_METRICS:
                boot[c][name][r] = bm[name]
        first = df.iloc[0]
        row = {
            "condition": c, "group": first.get("group"), "degradation": first.get("degradation"),
            "severity": first.get("severity"), "enhancement": first.get("enhancement"), "n": m["n"],
            **{name: m[name] for name in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                                           "macro_f1", "macro_specificity", "macro_roc_auc", "mean_confidence",
                                           "ece", "brier", "nll")},
            "mean_conf_correct": float(df["confidence"][y_pred == y_true].mean()) if (y_pred == y_true).any() else np.nan,
            "mean_conf_wrong": float(df["confidence"][y_pred != y_true].mean()) if (y_pred != y_true).any() else np.nan,
            "questionable_fraction": float((df.get("label_usefulness") == "questionable").mean())
            if "label_usefulness" in df else np.nan,
        }
        for name in BOOTSTRAP_METRICS:
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = bootstrap_ci(boot[c][name], level)
        summary_rows.append(row)
        for label in labels:
            pc = m["per_class"][label]
            per_class_rows.append({"condition": c, "label": label, **pc,
                                   "roc_auc": m["roc_auc_per_class"][label]})

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    vs_original = _paired_table(conditions, by_cond, preds, y_true, boot, labels, level, reference=BASELINE)
    restoration = _restoration_table(summary, by_cond, preds, y_true, boot, labels, level)
    severity = _severity_table(summary, primary)
    strata = _quality_strata(predictions)
    return EvaluationResult(summary, per_class, vs_original, restoration, severity, strata,
                            {"per_condition": details, "bootstrap": {"resamples": n_resamples, "level": level,
                                                                     "unit": "lesion" if "lesion_id" in base else "image",
                                                                     "n_groups": int(len(np.unique(groups)))}})


def _paired_row(c: str, ref: str, by_cond: dict[str, pd.DataFrame], preds: dict[str, np.ndarray],
                y_true: np.ndarray, boot: dict[str, dict[str, np.ndarray]], labels: list[str],
                level: float) -> dict[str, Any]:
    correct_c, correct_r = preds[c] == y_true, preds[ref] == y_true
    ref_only, cond_only, p = mcnemar_exact(correct_r, correct_c)
    true_idx = np.arange(len(y_true)), y_true
    _, probs_c = _arrays(by_cond[c], labels)
    _, probs_r = _arrays(by_cond[ref], labels)
    row: dict[str, Any] = {
        "condition": c, "reference": ref,
        "flip_rate": float((preds[c] != preds[ref]).mean()),
        "correct_to_wrong": ref_only, "wrong_to_correct": cond_only, "mcnemar_p": p,
        "delta_p_true_class": float((probs_c[true_idx] - probs_r[true_idx]).mean()),
        "delta_confidence": float(probs_c.max(axis=1).mean() - probs_r.max(axis=1).mean()),
    }
    k = len(labels)
    for name in BOOTSTRAP_METRICS:
        diffs = boot[c][name] - boot[ref][name]  # same resampled lesions → paired
        row[f"delta_{name}"] = _point(y_true, preds[c], k, name) - _point(y_true, preds[ref], k, name)
        row[f"delta_{name}_ci_low"], row[f"delta_{name}_ci_high"] = bootstrap_ci(diffs, level)
    return row


def _paired_table(conditions, by_cond, preds, y_true, boot, labels, level, reference) -> pd.DataFrame:
    rows = []
    for c in conditions:
        if c == reference:
            continue
        rows.append(_paired_row(c, reference, by_cond, preds, y_true, boot, labels, level))
    table = pd.DataFrame(rows)
    if len(table):
        table["mcnemar_p_holm"] = holm_adjust(table["mcnemar_p"].tolist())
    return table


def _point(y_true: np.ndarray, y_pred: np.ndarray, k: int, name: str) -> float:
    return metrics_from_confusion(confusion(y_true, y_pred, k))[name]


def _restoration_table(summary, by_cond, preds, y_true, boot, labels, level) -> pd.DataFrame:
    """For each degrade+enhance condition: effect relative to its degraded input and to the original."""
    rows = []
    k = len(labels)
    for _, s in summary[summary["group"] == "degrade_enhance"].iterrows():
        degraded = f"{s['degradation']}__{s['severity']}"
        if degraded not in by_cond:
            log.warning("Restoration '%s' has no matching degraded condition '%s' — skipped", s["condition"], degraded)
            continue
        row = _paired_row(s["condition"], degraded, by_cond, preds, y_true, boot, labels, level)
        orig = _point(y_true, preds[BASELINE], k, "balanced_accuracy")
        deg = _point(y_true, preds[degraded], k, "balanced_accuracy")
        enh = _point(y_true, preds[s["condition"]], k, "balanced_accuracy")
        row.update(degraded_condition=degraded, enhancement=s["enhancement"],
                   balacc_original=orig, balacc_degraded=deg, balacc_enhanced=enh,
                   recovery_fraction=(enh - deg) / (orig - deg) if abs(orig - deg) > 1e-9 else np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table):
        table["mcnemar_p_holm"] = holm_adjust(table["mcnemar_p"].tolist())
    return table


def _severity_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    deg = summary[summary["group"] == "degradation"]
    if deg.empty:
        return pd.DataFrame()
    table = deg.pivot(index="degradation", columns="severity", values=metric)
    order = [s for s in ("mild", "moderate", "severe") if s in table.columns]
    table = table[order + [c for c in table.columns if c not in order]]
    table.insert(0, "original", float(summary.loc[summary["condition"] == BASELINE, metric].iloc[0]))
    return table


def _quality_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    """Accuracy split by the heuristic quality flag (ok vs questionable)."""
    if "label_usefulness" not in predictions or "correct" not in predictions:
        return pd.DataFrame()
    table = predictions.groupby(["condition", "label_usefulness"], sort=False)["correct"].agg(["mean", "size"])
    return table.unstack("label_usefulness").sort_index(axis=1)
