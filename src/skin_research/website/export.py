"""Export experiment outputs into presentation data for the static website.

Reads (never writes) the files a finished experiment produced:

* ``outputs/predictions/<exp>/{predictions.csv, run_info.json, status.json}``
* ``outputs/reports/<exp>/{metrics_summary.csv, metrics_vs_original.csv, metrics_per_class.csv,
  metrics_restoration.csv, metrics.json, report.md}``
* the generated variation images listed in ``predictions.csv`` (``output_path``)

and writes only into the website folder:

* ``<site>/data/site-data.js``  (``window.SITE_DATA = {...}``; loads from ``file://`` without a server)
* ``<site>/data/SAMPLES.md``    (the real image IDs used, why each was chosen, and data provenance)
* ``<site>/assets/samples/<image_id>/<condition>.jpg`` and ``thumb.jpg``

An experiment is exported as results only when it is verifiably complete; otherwise only its
status (not started / running with progress / failed / incomplete) is exported.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

from skin_research.config import AppConfig, find_project_root, load_config
from skin_research.data.metadata import LABEL_DESCRIPTIONS
from skin_research.experiment import ExperimentPaths
from skin_research.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_WEBSITE_CONFIG = "configs/website.yaml"
STALE_RUNNING_HOURS = 6


# --------------------------------------------------------------------------- helpers
def _clean(value: Any) -> Any:
    """JSON-safe values: NaN/inf → None, numpy scalars → Python."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _num(x: Any, digits: int = 4) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, digits)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def _safe_name(condition: str) -> str:
    return condition.replace("+", "--")


@dataclass
class WebsiteConfig:
    root: Path
    output_dir: Path
    experiments: list[dict[str, str]]
    primary: str
    max_lesions: int = 8
    image_long_side: int = 520
    thumb_long_side: int = 160
    jpeg_quality: int = 86


def load_website_config(path: str | Path | None = None) -> WebsiteConfig:
    root = find_project_root()
    cfg_path = Path(path) if path else root / DEFAULT_WEBSITE_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    experiments = data.get("experiments") or []
    if not experiments:
        raise ValueError(f"{cfg_path}: 'experiments' must list at least one experiment.")
    keys = [e["key"] for e in experiments]
    primary = data.get("primary", keys[0])
    if primary not in keys:
        raise ValueError(f"{cfg_path}: primary '{primary}' is not one of {keys}.")
    samples = data.get("samples") or {}
    return WebsiteConfig(root=root, output_dir=root / data.get("output_dir", "website"), experiments=experiments,
                         primary=primary, max_lesions=int(samples.get("max_lesions", 8)),
                         image_long_side=int(samples.get("image_long_side", 520)),
                         thumb_long_side=int(samples.get("thumb_long_side", 160)),
                         jpeg_quality=int(samples.get("jpeg_quality", 86)))


# --------------------------------------------------------------------------- status
def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _metadata_rows(cfg: AppConfig) -> int | None:
    try:
        return int(len(pd.read_csv(cfg.dataset.metadata_csv)))
    except (OSError, ValueError):
        return None


def experiment_status(cfg: AppConfig) -> dict[str, Any]:
    """Decide whether an experiment's outputs are complete and trustworthy.

    ``complete`` requires: no running/failed status, run_info + predictions + metric tables present,
    prediction count == images × conditions, every condition evaluated, and (when no subsample was
    requested) every image in the dataset evaluated.
    """
    paths = ExperimentPaths.for_config(cfg)
    rep_dir = cfg.paths.reports_dir / cfg.experiment.name
    base = {"experiment": cfg.experiment.name, "dataset": cfg.dataset.name}
    status: dict[str, Any] = {}
    if paths.status.is_file():
        try:
            status = json.loads(paths.status.read_text())
        except (OSError, json.JSONDecodeError):
            status = {}
    state = status.get("state")
    if state == "running":
        updated = status.get("updated_utc")
        age_h = None
        if updated:
            age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(updated)).total_seconds() / 3600
        alive = _pid_alive(status.get("pid"))
        if not alive and age_h is not None and age_h > 0.25:
            return {**base, "state": "interrupted", "detail": "A run started but its process is no longer active.",
                    "done": status.get("done"), "total": status.get("total"), "updated_utc": updated}
        return {**base, "state": "running", "stage": status.get("stage"), "done": status.get("done"),
                "total": status.get("total"), "updated_utc": updated,
                "stale": bool(age_h and age_h > STALE_RUNNING_HOURS)}
    if state == "failed":
        return {**base, "state": "failed", "detail": status.get("error"), "updated_utc": status.get("updated_utc")}

    needed = [paths.run_info, paths.predictions, rep_dir / "metrics_summary.csv",
              rep_dir / "metrics_vs_original.csv", rep_dir / "metrics_per_class.csv", rep_dir / "metrics.json"]
    if not any(p.is_file() for p in needed):
        prepared = cfg.dataset.metadata_csv.is_file()
        return {**base, "state": "not_started" if prepared else "not_prepared",
                "detail": None if prepared else f"{_rel(cfg.dataset.metadata_csv, cfg.root)} does not exist yet."}
    missing = [_rel(p, cfg.root) for p in needed if not p.is_file()]
    if missing:
        return {**base, "state": "incomplete", "detail": "Missing: " + ", ".join(missing)}

    run_info = json.loads(paths.run_info.read_text())
    n_img, n_cond, n_pred = run_info.get("n_images"), run_info.get("n_conditions"), run_info.get("n_predictions")
    problems = []
    if not (n_img and n_cond and n_pred == n_img * n_cond):
        problems.append(f"prediction count {n_pred} ≠ images {n_img} × conditions {n_cond}")
    n_summary = len(pd.read_csv(rep_dir / "metrics_summary.csv"))
    if n_summary != n_cond:
        problems.append(f"{n_summary} evaluated conditions, expected {n_cond}")
    n_rows = sum(1 for _ in paths.predictions.open(encoding="utf-8")) - 1
    if n_rows != n_pred:
        problems.append(f"predictions.csv has {n_rows} rows, run_info says {n_pred}")
    expected_imgs = _metadata_rows(cfg)
    if cfg.dataset.max_images is None and expected_imgs and n_img != expected_imgs:
        problems.append(f"evaluated {n_img} of {expected_imgs} dataset images (a partial or trial run)")
    if run_info.get("dataset", {}).get("name") != cfg.dataset.name:
        problems.append("outputs belong to a different dataset")
    if problems:
        return {**base, "state": "incomplete", "detail": "; ".join(problems)}
    return {**base, "state": "complete", "n_images": n_img, "n_conditions": n_cond, "n_predictions": n_pred,
            "finished_utc": status.get("updated_utc") or run_info.get("environment", {}).get("timestamp_utc")}


# --------------------------------------------------------------------------- experiment results
def _observations(report_md: Path) -> list[str]:
    if not report_md.is_file():
        return []
    match = re.search(r"## 7\. Automated observations\n(.*?)\n## 8", report_md.read_text(encoding="utf-8"), re.S)
    return [ln[2:].strip() for ln in match.group(1).splitlines() if ln.startswith("- ")] if match else []


def _confidence_histograms(pred: pd.DataFrame, bins: int = 10) -> dict[str, dict[str, list[int]]]:
    out = {}
    edges = [i / bins for i in range(bins + 1)]
    for cond, df in pred.groupby("condition", sort=False):
        out[cond] = {}
        for key, part in (("correct", df[df["correct"]]), ("wrong", df[~df["correct"]])):
            counts = pd.cut(part["confidence"], edges, include_lowest=True).value_counts(sort=False)
            out[cond][key] = [int(v) for v in counts.values]
    return out


def export_experiment(cfg: AppConfig) -> dict[str, Any]:
    """All metrics of a complete experiment, read from its output files."""
    paths = ExperimentPaths.for_config(cfg)
    rep = cfg.paths.reports_dir / cfg.experiment.name
    root = cfg.root
    run_info = json.loads(paths.run_info.read_text())
    summary = pd.read_csv(rep / "metrics_summary.csv")
    vs = pd.read_csv(rep / "metrics_vs_original.csv").set_index("condition")
    per_class = pd.read_csv(rep / "metrics_per_class.csv")
    restoration = pd.read_csv(rep / "metrics_restoration.csv") if (rep / "metrics_restoration.csv").is_file() \
        else pd.DataFrame()
    details = json.loads((rep / "metrics.json").read_text()).get("per_condition", {})
    pred = pd.read_csv(paths.predictions, usecols=["condition", "confidence", "correct"])

    conditions = []
    for _, s in summary.iterrows():
        c = s["condition"]
        v = vs.loc[c] if c in vs.index else None
        row = {
            "name": c,
            "acc": _num(s["accuracy"]), "bal": _num(s["balanced_accuracy"]),
            "bal_lo": _num(s["balanced_accuracy_ci_low"]), "bal_hi": _num(s["balanced_accuracy_ci_high"]),
            "acc_lo": _num(s["accuracy_ci_low"]), "acc_hi": _num(s["accuracy_ci_high"]),
            "f1": _num(s["macro_f1"]), "auc": _num(s["macro_roc_auc"]),
            "spec": _num(s["macro_specificity"]), "conf": _num(s["mean_confidence"]),
            "conf_ok": _num(s["mean_conf_correct"]), "conf_bad": _num(s["mean_conf_wrong"]),
            "ece": _num(s["ece"]), "brier": _num(s["brier"]), "questionable": _num(s["questionable_fraction"]),
        }
        if v is not None:
            row.update(d_bal=_num(v["delta_balanced_accuracy"]), d_lo=_num(v["delta_balanced_accuracy_ci_low"]),
                       d_hi=_num(v["delta_balanced_accuracy_ci_high"]), flip=_num(v["flip_rate"]),
                       c2w=int(v["correct_to_wrong"]), w2c=int(v["wrong_to_correct"]),
                       p_holm=_num(v["mcnemar_p_holm"], 6), d_ptrue=_num(v["delta_p_true_class"]))
        conditions.append(row)

    pc: dict[str, dict[str, dict[str, Any]]] = {}
    for _, r in per_class.iterrows():
        pc.setdefault(r["condition"], {})[r["label"]] = {
            k: _num(r[k]) for k in ("precision", "recall", "specificity", "f1", "roc_auc")} | {
            "support": int(r["support"]) if not pd.isna(r["support"]) else 0}

    resto = [{
        "name": r["condition"], "degraded": r["degraded_condition"],
        "orig": _num(r["balacc_original"]), "deg": _num(r["balacc_degraded"]), "enh": _num(r["balacc_enhanced"]),
        "d": _num(r["delta_balanced_accuracy"]), "lo": _num(r["delta_balanced_accuracy_ci_low"]),
        "hi": _num(r["delta_balanced_accuracy_ci_high"]), "recovery": _num(r["recovery_fraction"]),
        "p_holm": _num(r["mcnemar_p_holm"], 6),
    } for _, r in restoration.iterrows()]

    ds = run_info.get("dataset", {})
    stats = ds.get("evaluation_split_stats", {})
    model = run_info.get("model", {})
    env = run_info.get("environment", {})
    return {
        "meta": {
            "experiment": cfg.experiment.name, "dataset": cfg.dataset.name,
            "n_images": run_info.get("n_images"), "n_conditions": run_info.get("n_conditions"),
            "n_predictions": run_info.get("n_predictions"), "n_lesions": stats.get("n_lesions"),
            "class_counts": stats.get("class_counts", {}), "split_level": ds.get("split_level"),
            "validation": ds.get("validation", {}),
            "inference_seconds": run_info.get("inference_seconds"),
            "throughput": run_info.get("inference_images_per_second"), "total_seconds": run_info.get("total_seconds"),
            "timestamp_utc": env.get("timestamp_utc"), "git_commit": env.get("git_commit"),
            "python": env.get("python"), "packages": env.get("packages", {}), "seed": run_info.get("seed"),
            "bootstrap": cfg.evaluation.bootstrap_resamples, "confidence_level": cfg.evaluation.confidence_level,
            "model": {k: model.get(k) for k in ("repo_id", "revision", "architecture", "parameters", "input_size",
                                                "license", "weights_sha256", "source_url", "reported_evaluation",
                                                "normalization", "labels")},
        },
        "conditions": conditions,
        "per_class": pc,
        "confusion": {c: d.get("confusion_matrix") for c, d in details.items()},
        "reliability": {c: d.get("reliability_bins") for c, d in details.items()},
        "confidence_hist": _confidence_histograms(pred),
        "restoration": resto,
        "observations": _observations(rep / "report.md"),
        "sources": {
            "summary": _rel(rep / "metrics_summary.csv", root),
            "vs_original": _rel(rep / "metrics_vs_original.csv", root),
            "per_class": _rel(rep / "metrics_per_class.csv", root),
            "restoration": _rel(rep / "metrics_restoration.csv", root),
            "details": _rel(rep / "metrics.json", root),
            "predictions": _rel(paths.predictions, root),
            "run_info": _rel(paths.run_info, root),
            "report": _rel(rep / "report.md", root),
        },
    }


# --------------------------------------------------------------------------- curated samples
def _wide(pred: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    pred = pred.copy()
    idx = {lbl: i for i, lbl in enumerate(labels)}
    probs = pred[[f"prob_{lbl}" for lbl in labels]].to_numpy()
    pred["p_true"] = [probs[i, idx[lbl]] for i, lbl in enumerate(pred["label"])]
    return pred


STORY_RULES: list[dict[str, Any]] = [
    {"key": "correct", "title": "Correct on the original photo",
     "what": "The model identifies this lesion correctly from the unaltered dermoscopic photo.",
     "condition": "original", "need": "orig_correct", "rank": "orig_conf"},
    {"key": "blur", "title": "Blur changes the diagnosis",
     "what": "Correct on the original, wrong after Gaussian blur (an out-of-focus photo).",
     "condition": ["gaussian_blur__mild", "gaussian_blur__moderate"], "need": "flip_to_wrong", "rank": "p_drop"},
    {"key": "dark", "title": "Poor lighting changes the diagnosis",
     "what": "Correct on the original, wrong when the photo is underexposed (poor lighting).",
     "condition": ["underexposure__moderate", "underexposure__mild"], "need": "flip_to_wrong", "rank": "p_drop"},
    {"key": "fixed", "title": "Brightness correction rescues it",
     "what": "Wrong when underexposed, correct again after brightness correction.",
     "condition": ["underexposure__moderate+brightness_correction"], "need": "rescued",
     "damaged": "underexposure__moderate", "rank": "p_gain"},
    {"key": "enhance_hurts", "title": "Enhancement makes it worse",
     "what": "Correct on the original, wrong after an enhancement was applied to the good photo.",
     "condition": ["enhance__clahe", "enhance__contrast_stretch", "enhance__denoise"], "need": "flip_to_wrong",
     "rank": "p_drop"},
    {"key": "confident_wrong", "title": "Confident, but wrong",
     "what": "Correct on the original; after degradation the model gives a wrong answer with high confidence.",
     "condition": "group:degradation", "need": "flip_to_wrong", "rank": "conf"},
    {"key": "noise", "title": "Sensor noise or compression changes the diagnosis",
     "what": "Correct on the original, wrong after sensor noise or JPEG compression.",
     "condition": ["gaussian_noise__moderate", "jpeg_compression__moderate"], "need": "flip_to_wrong",
     "rank": "p_drop"},
    {"key": "robust", "title": "Robust to most damage",
     "what": "Stays correct under the largest number of conditions.",
     "condition": "original", "need": "orig_correct", "rank": "n_correct"},
]


def select_stories(pred: pd.DataFrame, labels: list[str], max_lesions: int,
                   preferred_label: str = "mel") -> tuple[list[dict[str, Any]], list[str]]:
    """Choose gallery cases by fixed rules applied to the real predictions.

    Each rule prefers a melanoma (the clinically most important class) and a lesion not
    already used. Rules with no qualifying image are reported, never filled with made-up cases.
    """
    w = _wide(pred, labels)
    by = {c: df.set_index("image_id") for c, df in w.groupby("condition")}
    orig = by["original"]
    n_correct = w.groupby("image_id")["correct"].sum()
    lesion_of = orig["lesion_id"].fillna(pd.Series(orig.index, index=orig.index)).astype(str).to_dict()
    used: list[str] = []
    used_lesions: set[str] = set()
    stories, missing = [], []
    for rule in STORY_RULES:
        conds = rule["condition"]
        if conds == "group:degradation":
            conds = sorted(w.loc[w["group"] == "degradation", "condition"].unique())
        ordered = rule["condition"] != "group:degradation"  # earlier listed conditions are preferred
        conds = [conds] if isinstance(conds, str) else conds
        candidates = []
        for ci, c in enumerate(conds):
            if c not in by:
                continue
            df = by[c].join(orig[["correct", "p_true", "confidence"]], rsuffix="_orig")
            if rule["need"] == "orig_correct":
                df = df[df["correct"]]
            elif rule["need"] == "flip_to_wrong":
                df = df[df["correct_orig"] & ~df["correct"]]
            elif rule["need"] == "wrong":
                df = df[~df["correct"]]
            elif rule["need"] == "rescued":
                dmg = by.get(rule["damaged"])
                if dmg is None:
                    continue
                df = df.join(dmg[["correct", "p_true"]], rsuffix="_dmg")
                df = df[df["correct"] & ~df["correct_dmg"]]
            score = {
                "orig_conf": df["confidence"], "p_drop": df["p_true_orig"] - df["p_true"],
                "p_gain": df["p_true"] - df.get("p_true_dmg", 0), "conf": df["confidence"],
                "n_correct": n_correct.reindex(df.index),
            }[rule["rank"]]
            for image_id, sc in score.items():
                candidates.append((image_id, c, float(sc), df.loc[image_id, "label"], ci if ordered else 0))
        if not candidates:
            missing.append(rule["key"])
            continue
        # prefer: a lesion not shown yet, a melanoma, the rule's first-listed condition, the strongest effect
        candidates.sort(key=lambda t: (lesion_of.get(t[0], t[0]) in used_lesions, t[3] != preferred_label,
                                       t[4], -t[2]))
        if rule["key"] == "correct" and any(t[0] == "ISIC_0034329" for t in candidates):
            pick = next(t for t in candidates if t[0] == "ISIC_0034329")  # same lesion as the `demo` command
        else:
            pick = candidates[0]
        image_id, cond, score, label, _ = pick
        if image_id not in used:
            if len(used) >= max_lesions:
                missing.append(rule["key"])
                continue
            used.append(image_id)
            used_lesions.add(lesion_of.get(image_id, image_id))
        stories.append({"key": rule["key"], "title": rule["title"], "what": rule["what"], "image_id": image_id,
                        "label": label, "condition": cond, "score": round(score, 4)})
    return stories, missing


def _write_jpeg(src: Path, dest: Path, long_side: int, quality: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
        im.save(dest, format="JPEG", quality=quality, optimize=True, progressive=True)


def export_samples(cfg: AppConfig, wcfg: WebsiteConfig) -> dict[str, Any]:
    paths = ExperimentPaths.for_config(cfg)
    labels = list(cfg.dataset.labels)
    pred = pd.read_csv(paths.predictions)
    stories, missing = select_stories(pred, labels, wcfg.max_lesions)
    lesion_ids = list(dict.fromkeys(s["image_id"] for s in stories))
    asset_root = wcfg.output_dir / "assets" / "samples"
    if asset_root.exists():
        shutil.rmtree(asset_root)  # generated presentation assets only
    lesions = []
    quality_cols = {"brightness": "brightness", "rms_contrast": "contrast", "laplacian_var": "blur",
                    "noise_sigma": "noise", "dark_clipped": "dark", "bright_clipped": "bright"}
    for image_id in lesion_ids:
        rows = pred[pred["image_id"] == image_id]
        first = rows.iloc[0]
        per = {}
        for _, r in rows.iterrows():
            src = Path(r["output_path"]) if isinstance(r["output_path"], str) else None
            rel = None
            if src is not None and src.is_file():
                rel = f"assets/samples/{image_id}/{_safe_name(r['condition'])}.jpg"
                _write_jpeg(src, wcfg.output_dir / rel, wcfg.image_long_side, wcfg.jpeg_quality)
                if r["condition"] == "original":
                    _write_jpeg(src, wcfg.output_dir / f"assets/samples/{image_id}/thumb.jpg",
                                wcfg.thumb_long_side, wcfg.jpeg_quality)
            per[r["condition"]] = {
                "img": rel, "pred": r["predicted_label"], "conf": _num(r["confidence"]),
                "margin": _num(r["margin"]), "correct": bool(r["correct"]),
                "probs": [_num(r[f"prob_{lbl}"]) for lbl in labels],
                "quality": {v: _num(r.get(k), 4) for k, v in quality_cols.items()},
                "flags": r["quality_flags"] if isinstance(r.get("quality_flags"), str) else "",
                "source": _rel(src, cfg.root) if src else None,
            }
        lesions.append({"id": image_id, "label": first["label"],
                        "lesion_id": first["lesion_id"] if isinstance(first.get("lesion_id"), str) else None,
                        "thumb": f"assets/samples/{image_id}/thumb.jpg", "conditions": per})
    return {"experiment": cfg.experiment.name, "stories": stories, "missing_stories": missing, "lesions": lesions,
            "source": _rel(paths.predictions, cfg.root)}


def condition_catalog(cfg: AppConfig) -> list[dict[str, Any]]:
    from skin_research.preprocessing.variations import build_conditions

    return [{"name": c.name, "group": c.group, "degradation": c.degradation, "severity": c.severity,
             "enhancement": c.enhancement, "degradation_params": c.degradation_params,
             "enhancement_params": c.enhancement_params} for c in build_conditions(cfg)]


# --------------------------------------------------------------------------- build
def build_website(website_config: str | Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Export all presentation data; returns a summary of what was written."""
    wcfg = load_website_config(website_config)
    if output_dir is not None:
        wcfg.output_dir = output_dir
    (wcfg.output_dir / "data").mkdir(parents=True, exist_ok=True)

    experiments: dict[str, Any] = {}
    configs: dict[str, AppConfig] = {}
    for exp in wcfg.experiments:
        cfg = load_config(wcfg.root / exp["config"])
        configs[exp["key"]] = cfg
        status = experiment_status(cfg)
        entry: dict[str, Any] = {"key": exp["key"], "label": exp["label"], "status": status}
        if status["state"] == "complete":
            entry["results"] = export_experiment(cfg)
        log.info("Experiment %-10s %s", exp["key"], status["state"])
        experiments[exp["key"]] = entry

    primary_cfg = configs[wcfg.primary]
    if experiments[wcfg.primary]["status"]["state"] != "complete":
        raise RuntimeError(f"The primary experiment '{wcfg.primary}' is not complete "
                           f"({experiments[wcfg.primary]['status']}). Run it first.")
    samples = export_samples(primary_cfg, wcfg)
    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary": wcfg.primary,
        "order": [e["key"] for e in wcfg.experiments],
        "experiments": experiments,
        "labels": list(primary_cfg.dataset.labels),
        "label_names": {lbl: LABEL_DESCRIPTIONS.get(lbl, lbl) for lbl in primary_cfg.dataset.labels},
        "catalog": condition_catalog(primary_cfg),
        "quality_flags": primary_cfg.quality.get("flags", {}),
        "samples": samples,
    }
    js = "/* Generated by `skin-research build-site`; do not edit. */\nwindow.SITE_DATA = " + \
        json.dumps(_clean(data), separators=(",", ":"), allow_nan=False) + ";\n"
    (wcfg.output_dir / "data" / "site-data.js").write_text(js, encoding="utf-8")
    _write_samples_md(wcfg, data)
    n_images = sum(1 for _ in (wcfg.output_dir / "assets" / "samples").rglob("*.jpg"))
    return {"output_dir": str(wcfg.output_dir), "experiments": {k: v["status"]["state"] for k, v in experiments.items()},
            "lesions": [l["id"] for l in samples["lesions"]], "stories": len(samples["stories"]),
            "missing_stories": samples["missing_stories"], "images": n_images,
            "data_kb": round((wcfg.output_dir / "data" / "site-data.js").stat().st_size / 1024)}


def _write_samples_md(wcfg: WebsiteConfig, data: dict[str, Any]) -> None:
    s = data["samples"]
    lines = ["# Images and data used by the website", "",
             f"Generated {data['generated_utc']} by `skin-research build-site`. Every image is a resized copy of a "
             f"real file listed in `{s['source']}` (experiment `{s['experiment']}`); predictions and probabilities "
             "are copied from that file unchanged.", "", "## Curated cases", "",
             "| Case | Image ID | Diagnosis | Condition shown | Rule |", "|---|---|---|---|---|"]
    for st in s["stories"]:
        lines.append(f"| {st['title']} | {st['image_id']} | {st['label']} | `{st['condition']}` | {st['what']} |")
    if s["missing_stories"]:
        lines += ["", "No qualifying image was found for: " + ", ".join(s["missing_stories"]) + "."]
    lines += ["", "## Lesions in the Image Lab", ""]
    for les in s["lesions"]:
        orig = les["conditions"]["original"]
        lines.append(f"- {les['id']} (diagnosis {les['label']}, lesion {les['lesion_id']}): original image "
                     f"`{orig['source']}`, {len(les['conditions'])} conditions")
    lines += ["", "## Experiments", ""]
    for key in data["order"]:
        e = data["experiments"][key]
        st = e["status"]
        lines.append(f"- {e['label']} (`{st['experiment']}`): **{st['state']}**"
                     + (f" — {st.get('detail')}" if st.get("detail") else ""))
        if "results" in e:
            for name, path in e["results"]["sources"].items():
                lines.append(f"  - {name}: `{path}`")
    (wcfg.output_dir / "data" / "SAMPLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
