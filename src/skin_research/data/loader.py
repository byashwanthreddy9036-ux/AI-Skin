"""Dataset loading, group-aware splitting and statistics.

Images are decoded lazily, one at a time, so memory stays flat regardless of
dataset size. Arrays are always ``uint8`` RGB with shape ``(H, W, 3)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from skin_research.config import AppConfig
from skin_research.data.metadata import read_metadata_csv
from skin_research.data.validation import ValidationReport, resolve_image_paths, validate_dataset
from skin_research.utils.logging import get_logger

log = get_logger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Read an image file as an RGB ``uint8`` array (EXIF orientation applied)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_image(image: np.ndarray, path: Path, fmt: str = "png") -> Path:
    """Save an RGB array; refuses to overwrite an existing file."""
    path = path.with_suffix(f".{fmt}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG" if fmt == "png" else fmt.upper())
    return path


def iter_images(paths: list[Path]) -> Iterator[tuple[Path, np.ndarray]]:
    for path in paths:
        yield path, load_image(path)


def choose_group_column(df: pd.DataFrame, preferred: str) -> tuple[str, str]:
    """Pick the identifier used for leakage-free splitting.

    Returns ``(column, level)``; level is ``lesion``, ``patient`` or ``image``.
    An identifier is only used when it is present for every row.
    """
    order = [preferred] + [c for c in ("patient_id", "lesion_id") if c != preferred]
    for column in order:
        if column in df.columns and df[column].notna().all():
            return column, column.removesuffix("_id")
    return "image_id", "image"


def assign_splits(df: pd.DataFrame, test_fraction: float, group_column: str, seed: int) -> pd.DataFrame:
    """Add a ``split`` column (``dev``/``test``), never splitting a group across both.

    With ``test_fraction == 1`` every row is ``test``. Otherwise
    :class:`~sklearn.model_selection.StratifiedGroupKFold` keeps class balance
    approximately equal while keeping all images of a group together; the dev
    fraction is therefore approximate (1/k for the nearest k).
    """
    out = df.copy()
    column, level = choose_group_column(out, group_column)
    out["split_group"] = out[column].astype(str)
    out.attrs["split_level"] = level
    if test_fraction >= 1.0:
        out["split"] = "test"
        return out
    k = max(2, round(1 / (1 - test_fraction)))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, dev_idx = next(splitter.split(out, out["label"], groups=out["split_group"]))
    out["split"] = "test"
    out.iloc[dev_idx, out.columns.get_loc("split")] = "dev"
    return out


def stratified_group_subsample(df: pd.DataFrame, n: int, group_column: str, seed: int) -> pd.DataFrame:
    """Select ~``n`` rows keeping class proportions and whole groups together.

    Every class present keeps at least one group, so rare classes are not lost.
    """
    if n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    groups = df.groupby(group_column)["label"].agg(lambda s: s.mode().iloc[0])
    sizes = df.groupby(group_column).size()
    chosen: list[str] = []
    for label, label_groups in groups.groupby(groups):
        quota = max(1, round(n * (df["label"] == label).sum() / len(df)))
        ids = label_groups.index.to_numpy()
        rng.shuffle(ids)
        count = 0
        for gid in ids:
            if count >= quota:
                break
            chosen.append(gid)
            count += sizes[gid]
    return df[df[group_column].isin(chosen)]


@dataclass
class PreparedDataset:
    """Validated evaluation set ready for the experiment runner."""

    table: pd.DataFrame              # valid rows of the evaluation split, with abs_path
    full: pd.DataFrame               # all rows with split assignment
    validation: ValidationReport
    split_level: str                 # lesion / patient / image

    @property
    def paths(self) -> list[Path]:
        return [Path(p) for p in self.table["abs_path"]]


def prepare_dataset(cfg: AppConfig, check_near_duplicates: bool = True) -> PreparedDataset:
    """Load metadata, validate it, assign splits and select the evaluation subset.

    The split is fixed (seeded) *before* any image variation is generated, so
    every condition is evaluated on exactly the same images.
    """
    ds = cfg.dataset
    df = read_metadata_csv(ds.metadata_csv)
    report = validate_dataset(df, ds.image_root, ds.labels, check_near_duplicates=check_near_duplicates)
    for issue in report.issues:
        log.warning(issue) if not issue.startswith("INFO") else log.info(issue)
    invalid = report.table[~report.table["is_valid"]]
    if len(invalid):
        log.warning("Excluding %d invalid row(s) from the experiment", len(invalid))

    full = assign_splits(report.table, ds.split.test_fraction, ds.split.group_column, cfg.seed)
    level = full.attrs.get("split_level", "image")
    if level == "image":
        log.warning("No complete lesion_id/patient_id column: splitting is image-level, so images of "
                    "the same lesion/patient may fall in different splits (leakage cannot be excluded).")
    table = full[(full["split"] == ds.split.evaluate_split) & full["is_valid"]]
    if ds.max_images:
        table = stratified_group_subsample(table, ds.max_images, "split_group", cfg.seed)
        log.info("Subsampled to %d images (max_images=%d)", len(table), ds.max_images)
    if table.empty:
        raise ValueError("No valid images in the evaluation split.")
    return PreparedDataset(table.reset_index(drop=True), full, report, level)


def dataset_statistics(df: pd.DataFrame, labels: list[str]) -> dict:
    """Class distribution and identifier coverage for a canonical metadata table."""
    counts = df["label"].value_counts()
    stats = {
        "n_images": len(df),
        "class_counts": {lbl: int(counts.get(lbl, 0)) for lbl in labels},
        "class_fractions": {lbl: round(float(counts.get(lbl, 0)) / max(len(df), 1), 4) for lbl in labels},
        "imbalance_ratio": round(float(counts.max() / max(counts[counts > 0].min(), 1)), 2) if len(counts) else None,
        "n_lesions": int(df["lesion_id"].nunique()) if df["lesion_id"].notna().any() else None,
        "n_patients": int(df["patient_id"].nunique()) if df["patient_id"].notna().any() else None,
    }
    if {"width", "height"} <= set(df.columns) and df["width"].notna().any():
        sizes = df["width"].astype("Int64").astype(str) + "x" + df["height"].astype("Int64").astype(str)
        stats["image_sizes"] = sizes.value_counts().head(5).to_dict()
    return stats


def load_metadata_with_paths(cfg: AppConfig) -> pd.DataFrame:
    df = read_metadata_csv(cfg.dataset.metadata_csv)
    df["abs_path"] = resolve_image_paths(df, cfg.dataset.image_root)
    return df
