"""Per-image quality assessment, heuristic quality flags and summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from skin_research.data.loader import load_image
from skin_research.quality.metrics import compute_metrics
from skin_research.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_FLAGS = {
    "min_laplacian_var": 15.0,
    "min_brightness": 0.12,
    "max_brightness": 0.92,
    "min_rms_contrast": 0.04,
    "max_clipped_fraction": 0.25,
    "min_short_side": 64,
}

METRIC_COLUMNS = [
    "brightness", "rms_contrast", "percentile_contrast", "laplacian_var", "tenengrad",
    "noise_sigma", "dark_clipped", "bright_clipped", "colorfulness",
]


def quality_flags(metrics: dict[str, float], height: int, width: int, thresholds: dict[str, float]) -> list[str]:
    """Return heuristic warnings. Flags mark images whose diagnostic usefulness is
    *questionable*; they are not validated clinical acceptance criteria."""
    t = {**DEFAULT_FLAGS, **thresholds}
    flags = []
    if metrics["laplacian_var"] < t["min_laplacian_var"]:
        flags.append("blurry")
    if metrics["brightness"] < t["min_brightness"]:
        flags.append("too_dark")
    if metrics["brightness"] > t["max_brightness"]:
        flags.append("too_bright")
    if metrics["rms_contrast"] < t["min_rms_contrast"]:
        flags.append("low_contrast")
    if max(metrics["dark_clipped"], metrics["bright_clipped"]) > t["max_clipped_fraction"]:
        flags.append("clipped")
    if min(height, width) < t["min_short_side"]:
        flags.append("low_resolution")
    return flags


def assess_array(image: np.ndarray, quality_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Quality metrics + flags for an in-memory RGB image."""
    quality_cfg = quality_cfg or {}
    h, w = image.shape[:2]
    metrics = compute_metrics(image, int(quality_cfg.get("analysis_long_side", 512)))
    flags = quality_flags(metrics, h, w, quality_cfg.get("flags", {}))
    return {"width": w, "height": h, "megapixels": round(h * w / 1e6, 4), **metrics,
            "quality_flags": ";".join(flags), "label_usefulness": "questionable" if flags else "ok"}


def assess_file(path: Path, quality_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Quality record for an image file, including file-level facts (size, format, DPI)."""
    record: dict[str, Any] = {"path": str(path), "valid": False, "error": None}
    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            record.update(format=im.format, mode=im.mode,
                          dpi=f"{round(float(dpi[0]))}x{round(float(dpi[1]))}" if dpi else None)
        record["file_size_kb"] = round(path.stat().st_size / 1024, 1)
        record.update(assess_array(load_image(path), quality_cfg))
        record["valid"] = True
    except (OSError, ValueError) as exc:
        record["error"] = str(exc)
    return record


def assess_paths(paths: list[Path], quality_cfg: dict[str, Any] | None = None,
                 ids: list[str] | None = None) -> pd.DataFrame:
    """Assess many images sequentially (one image in memory at a time)."""
    rows = []
    for i, path in enumerate(paths):
        row = assess_file(Path(path), quality_cfg)
        if ids is not None:
            row = {"image_id": ids[i], **row}
        rows.append(row)
        if (i + 1) % 100 == 0:
            log.info("  assessed %d/%d images", i + 1, len(paths))
    return pd.DataFrame(rows)


def summarize_quality(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean/std/median (or per-group means) of each metric."""
    cols = [c for c in METRIC_COLUMNS if c in df.columns]
    if by:
        return df.groupby(by, sort=False)[cols].mean()
    return df[cols].describe(percentiles=[0.05, 0.5, 0.95]).T


def flag_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in df.get("quality_flags", pd.Series(dtype=str)).fillna(""):
        for flag in filter(None, str(value).split(";")):
            counts[flag] = counts.get(flag, 0) + 1
    return counts
