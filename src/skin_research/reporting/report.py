"""Automated Markdown research report.

The report separates three kinds of content explicitly:

* **Results** — numbers computed by this run (tables are generated, not written);
* **Automated observations** — rule-based statements derived from those numbers
  (e.g. "the CI of the difference excludes 0"), worded conservatively;
* **Interpretation & limitations** — fixed caveats that apply to the design.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from skin_research.config import AppConfig
from skin_research.data.metadata import LABEL_DESCRIPTIONS
from skin_research.evaluation.comparison import EvaluationResult


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _table(df: pd.DataFrame, digits: int = 3, index: bool = False) -> str:
    return df.to_markdown(index=index, floatfmt=f".{digits}f") if len(df) else "_(none)_"


def _ci(row: pd.Series, metric: str) -> str:
    return f"{_fmt(row[metric])} [{_fmt(row[f'{metric}_ci_low'])}, {_fmt(row[f'{metric}_ci_high'])}]"


def _rel(path: str | Path, start: Path) -> str:
    return os.path.relpath(Path(path), start)


def _observations(result: EvaluationResult, primary: str, alpha: float = 0.05) -> list[str]:
    """Conservative, rule-based statements derived from the computed numbers."""
    obs: list[str] = []
    s = result.summary.set_index("condition")
    base = s.loc["original"]
    vs = result.vs_original.set_index("condition") if len(result.vs_original) else pd.DataFrame()
    key = f"delta_{primary}"

    def sig(row: pd.Series) -> bool:
        return row[f"{key}_ci_high"] < 0 or row[f"{key}_ci_low"] > 0

    obs.append(f"Baseline (original images): {primary.replace('_', ' ')} {_ci(base, primary)}, "
               f"accuracy {_fmt(base['accuracy'])}, macro F1 {_fmt(base['macro_f1'])}, "
               f"macro ROC-AUC {_fmt(base['macro_roc_auc'])}, ECE {_fmt(base['ece'])}.")
    if len(vs):
        worse = vs[(vs[f"{key}_ci_high"] < 0)].sort_values(key)
        better = vs[(vs[f"{key}_ci_low"] > 0)].sort_values(key, ascending=False)
        if len(worse):
            names = ", ".join(f"`{c}` ({_fmt(r[key], 3)})" for c, r in worse.head(8).iterrows())
            obs.append(f"{len(worse)} condition(s) have a 95% CI for the change in {primary.replace('_', ' ')} "
                       f"entirely below 0 (lower than original). Largest drops: {names}.")
        if len(better):
            names = ", ".join(f"`{c}` (+{_fmt(r[key], 3)})" for c, r in better.iterrows())
            obs.append(f"{len(better)} condition(s) have a CI entirely above 0 (higher than original): {names}.")
        else:
            obs.append("No condition has a 95% CI for the change that lies entirely above 0: no transformation "
                       "was shown to improve on the original images.")
        held = vs[(vs["mcnemar_p_holm"] < alpha)]
        obs.append(f"After Holm correction, {len(held)} of {len(vs)} comparisons with the original reach "
                   f"p < {alpha} on McNemar's exact test of per-image correctness.")
        enh = vs[[c.startswith("enhance__") for c in vs.index]]
        for c, r in enh.iterrows():
            verdict = "lower" if r[f"{key}_ci_high"] < 0 else "higher" if r[f"{key}_ci_low"] > 0 else \
                "not distinguishable from"
            obs.append(f"Enhancing good-quality originals with `{c.removeprefix('enhance__')}`: change "
                       f"{_fmt(r[key])} [{_fmt(r[f'{key}_ci_low'])}, {_fmt(r[f'{key}_ci_high'])}] → {verdict} "
                       f"original; predictions flipped for {100 * r['flip_rate']:.1f}% of images.")
    if len(result.severity):
        mono = []
        for deg, row in result.severity.iterrows():
            vals = [row[c] for c in ("original", "mild", "moderate", "severe") if c in row.index]
            if all(a >= b for a, b in zip(vals, vals[1:])):
                mono.append(deg)
        obs.append(f"Performance decreased monotonically with severity for {len(mono)} of "
                   f"{len(result.severity)} degradation types ({', '.join(mono) or 'none'}).")
    if len(result.restoration):
        r = result.restoration.set_index("condition")
        helped = r[r[f"{key}_ci_low"] > 0]
        hurt = r[r[f"{key}_ci_high"] < 0]
        obs.append(f"Enhancement after degradation: {len(helped)} of {len(r)} pairs improved on the degraded "
                   f"images with a CI excluding 0 ({', '.join(f'`{c}`' for c in helped.index) or 'none'}); "
                   f"{len(hurt)} made them worse ({', '.join(f'`{c}`' for c in hurt.index) or 'none'}).")
        full = r[(r["recovery_fraction"] >= 0.9)]
        if len(full):
            obs.append("Point estimates recovered ≥90% of the lost balanced accuracy for: "
                       + ", ".join(f"`{c}`" for c in full.index) + " (see CIs before generalising).")
    # Rare classes dominate macro metrics: say so with the actual counts
    pc = result.per_class
    support = pc[pc["condition"] == "original"].set_index("label")["support"]
    present = support[support > 0]
    rare = present[present < 5]
    if len(rare):
        obs.append(f"Rare classes ({', '.join(f'{k}: {int(v)} image(s)' for k, v in rare.items())}) each carry "
                   f"1/{len(present)} = {1 / len(present):.3f} of balanced accuracy and macro F1, so a single image "
                   "flipping moves these metrics noticeably. Read macro metrics together with plain accuracy and the "
                   "per-class table.")
    # Melanoma sensitivity is the clinically most consequential number
    if "mel" in set(pc["label"]):
        mel = pc[pc["label"] == "mel"].set_index("condition")["recall"]
        n_mel = int(support.get("mel", 0))
        if n_mel:
            worst = mel.drop("original").nsmallest(5)
            obs.append(f"Melanoma sensitivity ({n_mel} melanomas): {_fmt(mel['original'])} on originals; lowest under "
                       + ", ".join(f"`{c}` {_fmt(v)}" for c, v in worst.items())
                       + ". Missed melanomas are a false-negative cancer outcome, the costliest error type.")
    # Calibration direction and confidence as an error signal (derived, not boilerplate)
    gap_o = base["mean_conf_correct"] - base["mean_conf_wrong"]
    direction = "under-confident" if base["mean_confidence"] < base["accuracy"] else "over-confident"
    obs.append(f"On original images the model is {direction}: mean top-1 confidence {_fmt(base['mean_confidence'])} "
               f"vs accuracy {_fmt(base['accuracy'])} (ECE {_fmt(base['ece'])}).")
    conf = result.summary[["condition", "mean_conf_correct", "mean_conf_wrong"]].dropna()
    if len(conf):
        gap = (conf["mean_conf_correct"] - conf["mean_conf_wrong"])
        inverted = conf.loc[gap < 0, "condition"].tolist()
        obs.append(f"Correct predictions were {_fmt(gap_o)} more confident than wrong ones on originals and "
                   f"{_fmt(gap.mean())} on average across conditions; in {len(inverted)} of {len(conf)} conditions "
                   "wrong predictions were on average *more* confident than correct ones. Confidence is therefore a "
                   "weak signal for detecting degraded-image errors with this model.")
    # Do the heuristic quality flags catch the damaging conditions?
    damaging = s[(s.index != "original") & (s[primary] < base[primary] - 0.2)]
    missed = damaging[damaging["questionable_fraction"] < 0.1]
    if len(missed):
        obs.append(f"The heuristic quality flags missed {len(missed)} of {len(damaging)} conditions that cut "
                   f"{primary.replace('_', ' ')} by more than 0.2 (<10% of images flagged): "
                   + ", ".join(f"`{c}`" for c in missed.index) + ". Simple no-reference metrics did not detect "
                   "all quality problems that harmed the model.")
    return obs


LIMITATIONS = [
    "**Small evaluation set.** Rare classes have very few images, so their per-class metrics and the macro "
    "averages are unstable. Confidence intervals are reported for this reason; differences whose CIs overlap "
    "zero are not evidence of an effect.",
    "**Synthetic degradations.** Controlled transformations approximate, but are not identical to, real "
    "acquisition problems (real blur is spatially varying; real lighting changes colour as well as intensity).",
    "**Label persistence.** Ground-truth labels are kept for every transformed image. At severe settings an "
    "expert might no longer be able to make the diagnosis either; such images are flagged by heuristic quality "
    "thresholds (`label_usefulness = questionable`), which are not validated clinical criteria.",
    "**Single model.** Results describe one pretrained EfficientNet-B0 checkpoint; other architectures or "
    "models trained with augmentation may be more or less robust.",
    "**Dermoscopic images only.** The model and data are dermoscopic; results do not transfer to smartphone "
    "or clinical photographs, other diseases, or populations under-represented in ISIC (e.g. darker skin tones).",
    "**Multiple comparisons.** Many conditions are compared with one baseline; McNemar p-values are Holm-adjusted, "
    "and bootstrap CIs are per-comparison (not simultaneous).",
    "**Not a medical device.** This is an educational research prototype and must not be used for diagnosis.",
]


def write_report(cfg: AppConfig, predictions: pd.DataFrame, result: EvaluationResult,
                 run_info: dict[str, Any], outputs: dict[str, Any]) -> Path:
    """Write ``outputs/reports/<experiment>/report.md`` and return its path."""
    out_dir = cfg.paths.reports_dir / cfg.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    primary = cfg.evaluation.primary_metric
    s = result.summary
    model = run_info.get("model", {})
    ds = run_info.get("dataset", {})
    env = run_info.get("environment", {})
    charts = outputs.get("charts", {})
    L: list[str] = []
    add = L.append

    add(f"# Skin lesion image quality vs. classifier performance — `{cfg.experiment.name}`\n")
    add("> Educational research prototype. **Not for clinical use.** Sections marked *Results* contain numbers "
        "computed by this run; *Automated observations* are rule-based statements derived from them; "
        "*Limitations* are fixed caveats of the design.\n")
    add("**Research question:** does improving (or degrading) the quality of skin lesion images change the "
        "classification accuracy and reliability of an existing pretrained model?\n")

    add("## 1. Experiment configuration\n")
    rows = [("Experiment", cfg.experiment.name), ("Random seed", cfg.seed),
            ("Images evaluated", run_info.get("n_images")), ("Conditions", run_info.get("n_conditions")),
            ("Predictions", run_info.get("n_predictions")), ("Primary metric", primary),
            ("Bootstrap", f"{cfg.evaluation.bootstrap_resamples} resamples of "
                          f"{result.details['bootstrap']['unit']}s ({result.details['bootstrap']['n_groups']} groups), "
                          f"{int(cfg.evaluation.confidence_level * 100)}% percentile CIs"),
            ("Config snapshot", "`config_used.yaml` next to the predictions")]
    add(_table(pd.DataFrame(rows, columns=["Setting", "Value"])) + "\n")

    add("## 2. Dataset\n")
    stats = ds.get("evaluation_split_stats", {})
    add(f"- Dataset: **{cfg.dataset.name}** (`{_rel(cfg.dataset.metadata_csv, cfg.root)}`)")
    val = ds.get("validation", {})
    add(f"- Validation: {val.get('valid_images')}/{val.get('rows')} images valid; missing {val.get('missing_files')}, "
        f"unreadable {val.get('unreadable_images')}, invalid labels {val.get('invalid_labels')}, "
        f"exact duplicate groups {val.get('exact_duplicate_groups')}, near-duplicate pairs "
        f"{val.get('near_duplicate_pairs')}.")
    add(f"- Grouping for splits/bootstrap: **{ds.get('split_level')}-level**"
        + (" (patient identifiers are not public, so patient-level independence cannot be guaranteed)."
           if ds.get("split_level") != "patient" else "."))
    add(f"- Distinct lesions: {stats.get('n_lesions', '–')}; class imbalance ratio (largest/smallest): "
        f"{stats.get('imbalance_ratio', '–')}\n")
    counts = stats.get("class_counts", {})
    add(_table(pd.DataFrame([(k, LABEL_DESCRIPTIONS.get(k, ""), v) for k, v in counts.items()],
                            columns=["Label", "Diagnosis", "Images"])) + "\n")

    add("## 3. Model\n")
    rep = model.get("reported_evaluation") or {}
    rows = [("Source", f"[{model.get('repo_id')}]({model.get('source_url')}) @ `{str(model.get('revision'))[:12]}`"),
            ("Architecture", f"{model.get('architecture')} ({model.get('parameters'):,} parameters)"
             if model.get("parameters") else model.get("architecture")),
            ("Input", f"{model.get('input_size')}×{model.get('input_size')} RGB, ImageNet normalisation"),
            ("Labels (model order)", ", ".join(model.get("labels", []))),
            ("Weights SHA-256", f"`{model.get('weights_sha256')}`"),
            ("Licence", model.get("license")),
            ("Card-reported (internal lesion-disjoint dev split)",
             f"macro-F1 {_fmt(rep.get('macro_f1'))}, balanced acc {_fmt(rep.get('balanced_accuracy'))}, "
             f"macro AUROC {_fmt(rep.get('macro_auroc'))}" if rep else "–")]
    add(_table(pd.DataFrame(rows, columns=["Property", "Value"])) + "\n")
    add("The model was trained on the ISIC 2018 Task 3 training set (HAM10000). The ISIC 2018 Validation and "
        "Test images used here were not part of that training release.\n")

    add("## 4. Image quality of the evaluation images (Results)\n")
    orig = predictions[predictions["condition"] == "original"]
    qcols = [c for c in ("brightness", "rms_contrast", "laplacian_var", "tenengrad", "noise_sigma",
                         "dark_clipped", "bright_clipped") if c in orig]
    if qcols:
        add(_table(orig[qcols].describe(percentiles=[0.05, 0.5, 0.95]).T.drop(columns="count"), 3, index=True) + "\n")
    if "quality" in charts:
        add(f"![quality]({_rel(charts['quality'], out_dir)})\n")
    add("Quality metrics are heuristics: the Laplacian variance, for example, rises with noise and JPEG "
        "blocking, so it is used only to compare an image with its own transformed versions.\n")

    add("## 5. Experimental conditions\n")
    cond = pd.DataFrame(run_info.get("conditions", []))
    if len(cond):
        cond["parameters"] = cond.apply(lambda r: ", ".join(
            f"{k}={v}" for d in (r["degradation_params"], r["enhancement_params"]) for k, v in (d or {}).items()), axis=1)
        add(_table(cond[["name", "group", "degradation", "severity", "enhancement", "parameters"]].fillna("–")) + "\n")
    add("Share of images flagged as questionable quality per condition is in the overview table below "
        "(`questionable`).\n")

    add("## 6. Classification performance (Results)\n")
    overview = s[["condition", "n", "accuracy", primary, f"{primary}_ci_low", f"{primary}_ci_high", "macro_f1",
                  "macro_precision", "macro_recall", "macro_specificity", "macro_roc_auc", "mean_confidence",
                  "ece", "brier"]].copy()
    overview["questionable"] = s["questionable_fraction"].values
    add(_table(overview) + "\n")
    if "condition_overview" in charts:
        add(f"![overview]({_rel(charts['condition_overview'], out_dir)})\n")

    add("### 6.1 Original vs degraded / enhanced (paired comparison)\n")
    add(f"Δ = condition − original on the same images. CI: paired lesion bootstrap. McNemar: exact test on "
        f"per-image correctness, Holm-adjusted across {len(result.vs_original)} comparisons.\n")
    key = f"delta_{primary}"
    vs = result.vs_original[["condition", key, f"{key}_ci_low", f"{key}_ci_high", "delta_macro_f1", "flip_rate",
                             "correct_to_wrong", "wrong_to_correct", "delta_p_true_class", "mcnemar_p_holm"]]
    add(_table(vs) + "\n")

    if len(result.severity):
        add("### 6.2 Degradation severity\n")
        add(_table(result.severity, index=True) + "\n")
        if "severity" in charts:
            add(f"![severity]({_rel(charts['severity'], out_dir)})\n")

    add("### 6.3 Enhancement\n")
    if "enhancement" in charts:
        add(f"![enhancement]({_rel(charts['enhancement'], out_dir)})\n")
    if len(result.restoration):
        add("Enhancement applied **after** degradation, compared with the degraded images (Δ = enhanced − degraded):\n")
        add(_table(result.restoration[["condition", "balacc_original", "balacc_degraded", "balacc_enhanced", key,
                                       f"{key}_ci_low", f"{key}_ci_high", "recovery_fraction", "mcnemar_p_holm"]])
            + "\n")

    add("### 6.4 Class-wise performance\n")
    pc = result.per_class
    sens = pc.pivot(index="condition", columns="label", values="recall").reindex(s["condition"])[cfg.dataset.labels]
    add("Sensitivity (recall) per class:\n")
    add(_table(sens, index=True) + "\n")
    spec = pc[pc["condition"] == "original"].set_index("label")[["support", "precision", "recall", "specificity",
                                                                  "f1", "roc_auc"]]
    add("Original images, per class:\n")
    add(_table(spec, index=True) + "\n")
    for key_chart in ("classwise", "confusion_original"):
        if key_chart in charts:
            add(f"![{key_chart}]({_rel(charts[key_chart], out_dir)})\n")

    add("### 6.5 Prediction confidence and calibration\n")
    add(_table(s[["condition", "mean_confidence", "mean_conf_correct", "mean_conf_wrong", "ece", "brier", "nll"]])
        + "\n")
    for key_chart in ("confidence", "reliability"):
        if key_chart in charts:
            add(f"![{key_chart}]({_rel(charts[key_chart], out_dir)})\n")
    if len(result.quality_strata):
        add("Accuracy on images flagged `ok` vs `questionable` by the heuristic quality checks "
            "(`mean` = accuracy, `size` = images):\n")
        strata = result.quality_strata.copy()
        strata.columns = [f"{a}_{b}" for a, b in strata.columns]
        add(_table(strata, index=True) + "\n")
    if "example" in charts:
        add("Example lesion across conditions:\n")
        add(f"![example]({_rel(charts['example'], out_dir)})\n")

    add("## 7. Automated observations\n")
    add("_Generated by rules from the numbers above; they describe this sample and are not causal or clinical "
        "claims._\n")
    for o in _observations(result, primary):
        add(f"- {o}")
    add("")

    add("## 8. Limitations\n")
    for lim in LIMITATIONS:
        add(f"- {lim}")
    add("")

    add("## 9. Execution details\n")
    pk = env.get("packages", {})
    rows = [("Timestamp (UTC)", env.get("timestamp_utc")), ("Git commit", env.get("git_commit")),
            ("Python", env.get("python")), ("Platform", env.get("platform")),
            ("Device", model.get("device")), ("Batch size", cfg.model.batch_size),
            ("Inference throughput", f"{run_info.get('inference_images_per_second')} images/s"),
            ("Inference time", f"{run_info.get('inference_seconds')} s"),
            ("Packages", ", ".join(f"{k} {v}" for k, v in pk.items() if v))]
    add(_table(pd.DataFrame(rows, columns=["Item", "Value"])) + "\n")
    add("Reproduce with:\n\n```bash\nuv sync\nuv run skin-research download-data\nuv run skin-research prepare-data "
        "--fetch-lesion-ids\nuv run skin-research download-model\nuv run skin-research experiment\n```\n")
    add("Output files: `predictions.csv` (one row per image × condition, with probabilities), "
        "`metrics_*.csv`, `metrics.json`, charts folder, and the variation manifest that links every generated "
        "image to its source image and parameters.\n")

    path.write_text("\n".join(L), encoding="utf-8")
    return path
