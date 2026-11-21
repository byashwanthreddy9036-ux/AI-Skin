"""Static explainer website built from a finished experiment.

``build_site`` reads the saved predictions, metric tables and run metadata,
picks a few example lesions, writes small JPEG thumbnails of what the model saw
under every condition, and inlines all numbers into one self-contained HTML
page (``outputs/site/index.html``). Nothing is recomputed or invented: every
number on the page comes from the experiment's output files.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from skin_research.config import AppConfig
from skin_research.data.metadata import LABEL_DESCRIPTIONS
from skin_research.utils.logging import get_logger

log = get_logger(__name__)

TEMPLATE = Path(__file__).with_name("site_template.html")
THUMB_LONG_SIDE = 360
EXAMPLE_PRIORITY = ["mel", "bcc", "bkl", "nv"]
PREFERRED_EXAMPLE = "ISIC_0034329"  # the melanoma used by the `demo` command


def _clean(value: Any) -> Any:
    """JSON-safe scalars (NaN → None, numpy → Python)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _r(x: Any, digits: int = 4) -> Any:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)


def pick_examples(predictions: pd.DataFrame, n: int = 4) -> list[str]:
    """Lesions that are correct on the original and change most under the conditions."""
    orig = predictions[predictions["condition"] == "original"].set_index("image_id")
    good = orig[orig["correct"]]
    changes = (predictions.merge(orig[["predicted_label"]].rename(columns={"predicted_label": "orig_pred"}),
                                 left_on="image_id", right_index=True)
               .assign(changed=lambda d: d["predicted_label"] != d["orig_pred"])
               .groupby("image_id")["changed"].sum())
    chosen: list[str] = []
    if PREFERRED_EXAMPLE in good.index:
        chosen.append(PREFERRED_EXAMPLE)
    for label in EXAMPLE_PRIORITY:
        if len(chosen) >= n:
            break
        if any(good.loc[c, "label"] == label for c in chosen):
            continue
        ids = good[good["label"] == label].index
        if len(ids):
            chosen.append(changes.reindex(ids).fillna(0).idxmax())
    return chosen[:n]


def _thumbnail(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((THUMB_LONG_SIDE, THUMB_LONG_SIDE), Image.Resampling.LANCZOS)
        im.save(dest, format="JPEG", quality=86, optimize=True)


def _observations(report_md: Path) -> list[str]:
    if not report_md.is_file():
        return []
    text = report_md.read_text(encoding="utf-8")
    match = re.search(r"## 7\. Automated observations\n(.*?)\n## 8", text, re.S)
    if not match:
        return []
    return [line[2:].strip() for line in match.group(1).splitlines() if line.startswith("- ")]


STANDALONE_HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def build_site(cfg: AppConfig, out_dir: Path | None = None, standalone: bool = True) -> Path:
    """Build ``index.html`` + ``img/`` for the configured experiment; returns the HTML path.

    ``standalone`` wraps the page in a full HTML document so it opens directly from
    disk; ``False`` writes only the page content (for hosts that add their own
    document skeleton).
    """
    name = cfg.experiment.name
    pred_dir = cfg.paths.predictions_dir / name
    rep_dir = cfg.paths.reports_dir / name
    out_dir = out_dir or (cfg.paths.outputs_dir / "site")
    needed = [pred_dir / "predictions.csv", pred_dir / "run_info.json", rep_dir / "metrics_summary.csv",
              rep_dir / "metrics_vs_original.csv", rep_dir / "metrics_per_class.csv"]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Run `skin-research experiment` first; missing: " + ", ".join(missing))

    predictions = pd.read_csv(pred_dir / "predictions.csv")
    run_info = json.loads((pred_dir / "run_info.json").read_text())
    summary = pd.read_csv(rep_dir / "metrics_summary.csv")
    vs = pd.read_csv(rep_dir / "metrics_vs_original.csv").set_index("condition")
    per_class = pd.read_csv(rep_dir / "metrics_per_class.csv")
    restoration_path = rep_dir / "metrics_restoration.csv"
    restoration = pd.read_csv(restoration_path) if restoration_path.is_file() else pd.DataFrame()
    labels = list(cfg.dataset.labels)

    recipes = {c["name"]: c for c in run_info.get("conditions", [])}
    recall = per_class.pivot(index="condition", columns="label", values="recall")
    conditions = []
    for _, s in summary.iterrows():
        c = s["condition"]
        v = vs.loc[c] if c in vs.index else None
        rec = recipes.get(c, {})
        conditions.append({
            "name": c, "group": s["group"], "degradation": s.get("degradation"), "severity": s.get("severity"),
            "enhancement": s.get("enhancement"),
            "params": {**(rec.get("degradation_params") or {}), **(rec.get("enhancement_params") or {})},
            "acc": _r(s["accuracy"]), "bal": _r(s["balanced_accuracy"]),
            "bal_lo": _r(s["balanced_accuracy_ci_low"]), "bal_hi": _r(s["balanced_accuracy_ci_high"]),
            "f1": _r(s["macro_f1"]), "auc": _r(s["macro_roc_auc"]), "conf": _r(s["mean_confidence"]),
            "conf_ok": _r(s["mean_conf_correct"]), "conf_bad": _r(s["mean_conf_wrong"]), "ece": _r(s["ece"]),
            "questionable": _r(s["questionable_fraction"]),
            "d_bal": _r(v["delta_balanced_accuracy"]) if v is not None else 0.0,
            "d_lo": _r(v["delta_balanced_accuracy_ci_low"]) if v is not None else None,
            "d_hi": _r(v["delta_balanced_accuracy_ci_high"]) if v is not None else None,
            "flip": _r(v["flip_rate"]) if v is not None else 0.0,
            "p_holm": _r(v["mcnemar_p_holm"], 6) if v is not None else None,
            "recall": {lbl: _r(recall.loc[c, lbl]) if c in recall.index else None for lbl in labels},
        })

    resto = [{
        "name": r["condition"], "degraded": r["degraded_condition"], "enhancement": r["enhancement"],
        "orig": _r(r["balacc_original"]), "deg": _r(r["balacc_degraded"]), "enh": _r(r["balacc_enhanced"]),
        "lo": _r(r["delta_balanced_accuracy_ci_low"]), "hi": _r(r["delta_balanced_accuracy_ci_high"]),
        "recovery": _r(r["recovery_fraction"]),
    } for _, r in restoration.iterrows()]

    # Example lesions with thumbnails of every condition
    img_dir = out_dir / "img"
    examples = []
    for image_id in pick_examples(predictions):
        rows = predictions[predictions["image_id"] == image_id]
        per_cond = {}
        for _, r in rows.iterrows():
            src = Path(r["output_path"]) if isinstance(r.get("output_path"), str) else None
            rel = None
            if src is not None and src.is_file():
                rel = f"img/{image_id}/{r['condition']}.jpg"
                _thumbnail(src, out_dir / rel)
            per_cond[r["condition"]] = {
                "img": rel, "pred": r["predicted_label"], "conf": _r(r["confidence"]),
                "correct": bool(r["correct"]), "probs": [_r(r[f"prob_{lbl}"]) for lbl in labels],
                "brightness": _r(r.get("brightness")), "contrast": _r(r.get("rms_contrast")),
                "blur": _r(r.get("laplacian_var"), 1), "noise": _r(r.get("noise_sigma"), 3),
                "flags": r.get("quality_flags") if isinstance(r.get("quality_flags"), str) else "",
            }
        examples.append({"id": image_id, "label": rows["label"].iloc[0], "conditions": per_cond})

    model = run_info.get("model", {})
    ds = run_info.get("dataset", {})
    data = {
        "meta": {
            "experiment": name, "n_images": run_info.get("n_images"), "n_conditions": run_info.get("n_conditions"),
            "n_predictions": run_info.get("n_predictions"), "seconds": run_info.get("total_seconds"),
            "throughput": run_info.get("inference_images_per_second"),
            "timestamp": (run_info.get("environment") or {}).get("timestamp_utc"),
            "git_commit": (run_info.get("environment") or {}).get("git_commit"),
            "dataset": cfg.dataset.name, "split_level": ds.get("split_level"),
            "class_counts": (ds.get("evaluation_split_stats") or {}).get("class_counts", {}),
            "n_lesions": (ds.get("evaluation_split_stats") or {}).get("n_lesions"),
            "bootstrap": cfg.evaluation.bootstrap_resamples,
            "model": {k: model.get(k) for k in ("repo_id", "revision", "architecture", "parameters", "input_size",
                                                "license", "weights_sha256", "source_url")},
        },
        "labels": labels,
        "label_names": {lbl: LABEL_DESCRIPTIONS.get(lbl, lbl) for lbl in labels},
        "conditions": conditions,
        "restoration": resto,
        "examples": examples,
        "observations": _observations(rep_dir / "report.md"),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__SITE_DATA__*/null", json.dumps(_clean(data), separators=(",", ":")))
    if standalone:
        head, _, body = html.partition("</style>")
        html = STANDALONE_HEAD + head + "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / ("index.html" if standalone else "page.html")
    page.write_text(html, encoding="utf-8")
    n_imgs = sum(1 for _ in img_dir.rglob("*.jpg")) if img_dir.is_dir() else 0
    log.info("Site written to %s (%d thumbnails, %d conditions)", page, n_imgs, len(conditions))
    return page
