"""Dataset validation: files, images, labels, identifiers and duplicates.

Validation never modifies data. It returns a per-image table with an
``is_valid`` flag and a list of human-readable issues, so a run can either
exclude bad rows or stop, but never silently ignores them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from skin_research.utils.logging import get_logger

log = get_logger(__name__)

MIN_IMAGE_SIDE = 32
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ValidationReport:
    table: pd.DataFrame                     # one row per metadata row
    issues: list[str] = field(default_factory=list)
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    near_duplicate_pairs: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return int(self.table["is_valid"].sum())

    @property
    def ok(self) -> bool:
        return self.n_valid == len(self.table) and not any(i.startswith("ERROR") for i in self.issues)

    def summary(self) -> dict:
        t = self.table
        return {
            "rows": len(t),
            "valid_images": self.n_valid,
            "missing_files": int((~t["exists"]).sum()),
            "unreadable_images": int((t["exists"] & ~t["readable"]).sum()),
            "invalid_labels": int((~t["label_valid"]).sum()),
            "exact_duplicate_groups": len(self.exact_duplicate_groups),
            "near_duplicate_pairs": len(self.near_duplicate_pairs),
            "issues": self.issues,
        }


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash: a 64-bit perceptual fingerprint robust to resizing/recompression."""
    gray = np.asarray(image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS),
                      dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def inspect_image(path: Path) -> dict:
    """Open an image safely and return basic facts (never raises for bad files)."""
    info = {"exists": path.is_file(), "readable": False, "width": None, "height": None,
            "mode": None, "format": None, "sha256": None, "dhash": None, "error": None}
    if not info["exists"]:
        info["error"] = "file not found"
        return info
    try:
        with Image.open(path) as im:
            im.verify()  # detects truncated / corrupt files
        with Image.open(path) as im:
            im.load()
            info.update(width=im.width, height=im.height, mode=im.mode, format=im.format,
                        dhash=f"{dhash(im):016x}")  # hex string: pandas would turn ints into lossy floats
        info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        info["readable"] = True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        info["error"] = f"unreadable image: {exc}"
    return info


def resolve_image_paths(df: pd.DataFrame, image_root: Path) -> pd.Series:
    return df["image_path"].map(lambda p: Path(p) if Path(p).is_absolute() else image_root / p)


def validate_dataset(
    df: pd.DataFrame,
    image_root: Path,
    allowed_labels: list[str],
    near_duplicate_threshold: int = 4,
    check_near_duplicates: bool = True,
) -> ValidationReport:
    """Validate a canonical metadata table against the files on disk."""
    issues: list[str] = []
    table = df.copy()
    table["abs_path"] = resolve_image_paths(table, image_root)

    if table.empty:
        return ValidationReport(table.assign(exists=[], readable=[], label_valid=[], is_valid=[]),
                                ["ERROR: metadata contains no rows"])

    allowed = set(allowed_labels)
    table["label_valid"] = table["label"].isin(allowed)
    bad_labels = sorted(set(table.loc[~table["label_valid"], "label"].astype(str)))
    if bad_labels:
        issues.append(f"ERROR: {len(bad_labels)} unknown label value(s): {bad_labels[:10]} (allowed: {sorted(allowed)})")

    suffix_bad = ~table["abs_path"].map(lambda p: p.suffix.lower() in SUPPORTED_SUFFIXES)
    if suffix_bad.any():
        issues.append(f"WARNING: {int(suffix_bad.sum())} file(s) have unsupported extensions")

    for column in ("image_path", "image_id"):
        dup = table[column].duplicated(keep=False) & table[column].notna()
        if dup.any():
            issues.append(f"ERROR: {int(dup.sum())} rows share a duplicated {column}")

    for column in ("lesion_id", "patient_id"):
        present = table[column].notna().sum()
        if 0 < present < len(table):
            issues.append(f"WARNING: {column} is missing for {len(table) - present} of {len(table)} rows")

    facts = [inspect_image(p) for p in table["abs_path"]]
    facts_df = pd.DataFrame(facts, index=table.index)
    table = pd.concat([table, facts_df], axis=1)
    small = table["readable"] & ((table["width"].fillna(0) < MIN_IMAGE_SIDE) | (table["height"].fillna(0) < MIN_IMAGE_SIDE))
    table.loc[small, "error"] = f"image smaller than {MIN_IMAGE_SIDE}px"

    n_missing = int((~table["exists"]).sum())
    n_unreadable = int((table["exists"] & ~table["readable"]).sum())
    if n_missing:
        issues.append(f"ERROR: {n_missing} image file(s) not found, e.g. {table.loc[~table['exists'], 'image_path'].iloc[0]}")
    if n_unreadable:
        issues.append(f"ERROR: {n_unreadable} image file(s) could not be decoded")

    table["is_valid"] = table["exists"] & table["readable"] & table["label_valid"] & ~small

    # Exact duplicates: identical bytes under different names.
    exact_groups: list[list[str]] = []
    hashed = table[table["sha256"].notna()]
    for _, group in hashed.groupby("sha256"):
        if len(group) > 1:
            exact_groups.append(group["image_id"].astype(str).tolist())
    if exact_groups:
        issues.append(f"WARNING: {len(exact_groups)} group(s) of byte-identical images")

    near_pairs: list[tuple[str, str, int]] = []
    if check_near_duplicates and len(hashed) > 1:
        near_pairs = _near_duplicates(hashed, near_duplicate_threshold)
        if near_pairs:
            label_of = dict(zip(table["image_id"].astype(str), table["label"]))
            cross = [p for p in near_pairs if label_of.get(p[0]) != label_of.get(p[1])]
            issues.append(
                f"INFO: {len(near_pairs)} near-duplicate image pair(s) (dHash distance ≤ {near_duplicate_threshold}); "
                f"{len(cross)} with different labels. Same-lesion photos are expected in ISIC/HAM10000."
            )

    log.info("Validation: %d/%d rows valid", int(table["is_valid"].sum()), len(table))
    return ValidationReport(table, issues, exact_groups, near_pairs)


def _near_duplicates(hashed: pd.DataFrame, threshold: int) -> list[tuple[str, str, int]]:
    """Pairwise Hamming distances between dHashes (vectorised; fine up to ~20k images)."""
    ids = hashed["image_id"].astype(str).to_numpy()
    bits = np.array([[(h >> i) & 1 for i in range(64)] for h in hashed["dhash"].map(lambda x: int(x, 16))], dtype=np.uint8)
    pairs: list[tuple[str, str, int]] = []
    chunk = 512
    for start in range(0, len(bits), chunk):
        block = bits[start:start + chunk]
        dist = (block[:, None, :] != bits[None, :, :]).sum(axis=2)
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows, cols):
            i = start + r
            if c > i:
                pairs.append((ids[i], ids[c], int(dist[r, c])))
    return pairs

