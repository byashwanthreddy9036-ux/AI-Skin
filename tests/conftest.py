"""Shared fixtures: a tiny synthetic 'dataset' so unit tests need no downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skin_research.config import config_from_dict, load_config

LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def synthetic_lesion(seed: int, size: tuple[int, int] = (120, 160)) -> np.ndarray:
    """Skin-coloured background with a dark textured blob (an image with real edges)."""
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = (220, 170, 150)
    cy, cx = h / 2 + rng.uniform(-10, 10), w / 2 + rng.uniform(-10, 10)
    mask = ((yy - cy) / (h * 0.3)) ** 2 + ((xx - cx) / (w * 0.25)) ** 2 < 1
    img[mask] = (110, 70, 50)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_image() -> np.ndarray:
    return synthetic_lesion(0)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """14 images (2 per class, 1 lesion each) + canonical metadata CSV."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for i in range(14):
        name = f"IMG_{i:03d}.jpg"
        Image.fromarray(synthetic_lesion(i)).save(image_root / name, quality=95)
        rows.append({"image_path": name, "label": LABELS[i % 7], "image_id": f"IMG_{i:03d}",
                     "lesion_id": f"L{i // 2}", "patient_id": ""})
    csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, image_root


@pytest.fixture
def tmp_config(tmp_path: Path, tiny_dataset):
    """Project config redirected to the synthetic dataset and a temp output tree."""
    csv, image_root = tiny_dataset
    raw = load_config().raw
    raw["dataset"]["metadata_csv"] = str(csv)
    raw["dataset"]["image_root"] = str(image_root)
    raw["paths"] = {k: str(tmp_path / k) for k in raw["paths"]}
    raw["evaluation"]["bootstrap_resamples"] = 50
    cfg = config_from_dict(raw, root=tmp_path)
    cfg.model.label_map = dict(load_config().model.label_map)
    return cfg
