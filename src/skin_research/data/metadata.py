"""Canonical metadata schema and converters for public skin lesion datasets.

Every dataset is converted to one CSV format so the rest of the system never
needs dataset-specific code:

=============  ========  ===================================================
column         required  meaning
=============  ========  ===================================================
image_path     yes       path relative to ``dataset.image_root`` (or absolute)
label          yes       canonical diagnosis code (``akiec, bcc, ..., vasc``)
image_id       no        unique image identifier (defaults to file stem)
lesion_id      no        lesion identifier → lesion-level splitting
patient_id     no        patient identifier → patient-level splitting
=============  ========  ===================================================
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from skin_research.utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("image_path", "label")
OPTIONAL_COLUMNS = ("image_id", "lesion_id", "patient_id")

# ISIC 2018 Task 3 ground-truth columns → canonical HAM10000 ``dx`` codes.
ISIC2018_COLUMNS = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
}

LABEL_DESCRIPTIONS = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma (Bowen's disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

ISIC_API_IMAGE_URL = "https://api.isic-archive.com/api/v2/images/{isic_id}/"


def read_metadata_csv(path: Path) -> pd.DataFrame:
    """Read a canonical metadata CSV and check the required columns exist."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {path}\n"
            "Run `skin-research download-data` then `skin-research prepare-data`, "
            "or point dataset.metadata_csv at your own CSV (see README 'Dataset setup')."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata CSV {path} is missing required columns: {missing}")
    df["label"] = df["label"].str.strip().str.lower()
    if "image_id" not in df.columns:
        df["image_id"] = df["image_path"].map(lambda p: Path(p).stem)
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def convert_isic2018_groundtruth(
    groundtruth_csv: Path, image_suffix: str = ".jpg", lesion_groupings_csv: Path | None = None
) -> pd.DataFrame:
    """Convert an ISIC 2018 Task 3 one-hot ground-truth CSV to the canonical format."""
    gt = pd.read_csv(groundtruth_csv)
    if "image" not in gt.columns or not set(ISIC2018_COLUMNS) <= set(gt.columns):
        raise ValueError(f"{groundtruth_csv} is not an ISIC 2018 Task 3 ground-truth file.")
    onehot = gt[list(ISIC2018_COLUMNS)].to_numpy()
    if not ((onehot.sum(axis=1) == 1).all()):
        raise ValueError("ISIC ground truth rows must contain exactly one positive class.")
    labels = [ISIC2018_COLUMNS[c] for c in gt[list(ISIC2018_COLUMNS)].idxmax(axis=1)]
    out = pd.DataFrame(
        {
            "image_path": gt["image"] + image_suffix,
            "label": labels,
            "image_id": gt["image"],
            "lesion_id": pd.NA,
            "patient_id": pd.NA,
        }
    )
    if lesion_groupings_csv is not None and lesion_groupings_csv.is_file():
        groups = pd.read_csv(lesion_groupings_csv)
        out["lesion_id"] = out["image_id"].map(groups.set_index("image")["lesion_id"])
        log.info("Attached lesion_id for %d/%d images", out["lesion_id"].notna().sum(), len(out))
    return out


def convert_ham10000_metadata(metadata_csv: Path, image_suffix: str = ".jpg") -> pd.DataFrame:
    """Convert ``HAM10000_metadata.csv`` (Harvard Dataverse / Kaggle) to canonical format."""
    meta = pd.read_csv(metadata_csv)
    required = {"lesion_id", "image_id", "dx"}
    if not required <= set(meta.columns):
        raise ValueError(f"{metadata_csv} lacks HAM10000 columns {sorted(required - set(meta.columns))}.")
    return pd.DataFrame(
        {
            "image_path": meta["image_id"] + image_suffix,
            "label": meta["dx"].str.lower(),
            "image_id": meta["image_id"],
            "lesion_id": meta["lesion_id"],
            "patient_id": pd.NA,  # HAM10000 does not publish patient identifiers
        }
    )


def _fetch_one(image_id: str, timeout_s: float, retries: int = 3) -> dict[str, str | None] | None:
    """One ISIC API lookup with retries; ``None`` if every attempt failed."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(ISIC_API_IMAGE_URL.format(isic_id=image_id),
                                         headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                clinical = json.load(resp).get("metadata", {}).get("clinical", {})
            return {"lesion_id": clinical.get("lesion_id"), "patient_id": clinical.get("patient_id")}
        # OSError covers URLError, timeouts, connection resets; HTTPException covers dropped connections
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            log.warning("ISIC API lookup for %s failed (attempt %d/%d): %s", image_id, attempt, retries, exc)
            time.sleep(2 ** attempt)
    return None


def _save_cache(cache_csv: Path, cached: dict[str, dict[str, str | None]]) -> None:
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_csv.with_suffix(".tmp")
    pd.DataFrame([{"image_id": k, **v} for k, v in cached.items()]).to_csv(tmp, index=False)
    tmp.replace(cache_csv)


def fetch_isic_lesion_ids(
    image_ids: list[str], cache_csv: Path, delay_s: float = 0.1, timeout_s: float = 20, save_every: int = 50
) -> pd.DataFrame:
    """Look up ``lesion_id`` (and ``patient_id`` if public) via the ISIC Archive API.

    Successful lookups are cached in ``cache_csv`` (saved every ``save_every`` lookups, so an
    interruption loses little) and never repeated. Failed lookups are retried, logged and left
    empty; they are not cached, so a later run tries them again. Network errors never abort.
    """
    cached: dict[str, dict[str, str | None]] = {}
    if cache_csv.is_file():
        prev = pd.read_csv(cache_csv, dtype=str, keep_default_na=False)
        cached = {r.image_id: {"lesion_id": r.lesion_id or None, "patient_id": r.patient_id or None}
                  for r in prev.itertuples()}
    todo = [i for i in image_ids if i not in cached]
    log.info("ISIC API lookup: %d cached, %d to fetch", len(image_ids) - len(todo), len(todo))
    failed = []
    for n, image_id in enumerate(todo, 1):
        record = _fetch_one(image_id, timeout_s)
        if record is None:
            failed.append(image_id)
        else:
            cached[image_id] = record
        if n % 25 == 0:
            log.info("  fetched %d/%d", n, len(todo))
        if n % save_every == 0:
            _save_cache(cache_csv, cached)
        time.sleep(delay_s)
    _save_cache(cache_csv, cached)
    if failed:
        log.warning("%d lookup(s) failed and were left empty (rerun to retry): %s", len(failed), failed[:10])
    empty = {"lesion_id": None, "patient_id": None}
    return pd.DataFrame([{"image_id": i, **cached.get(i, empty)} for i in image_ids])
