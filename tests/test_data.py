from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skin_research.data.download import safe_extract
from skin_research.data.loader import (
    assign_splits,
    load_image,
    prepare_dataset,
    save_image,
    stratified_group_subsample,
)
from skin_research.data.metadata import convert_isic2018_groundtruth, read_metadata_csv
from skin_research.data.validation import validate_dataset

from .conftest import LABELS


def test_load_dataset(tiny_dataset):
    csv, root = tiny_dataset
    df = read_metadata_csv(csv)
    assert len(df) == 14 and set(df["label"]) == set(LABELS)
    img = load_image(root / df["image_path"][0])
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3


def test_validation_detects_missing_corrupt_and_bad_labels(tiny_dataset):
    csv, root = tiny_dataset
    df = read_metadata_csv(csv)
    (root / "broken.jpg").write_bytes(b"not an image")
    extra = pd.DataFrame([
        {"image_path": "missing.jpg", "label": "mel", "image_id": "X1"},
        {"image_path": "broken.jpg", "label": "nv", "image_id": "X2"},
        {"image_path": df["image_path"][0], "label": "psoriasis", "image_id": "X3"},
    ])
    report = validate_dataset(pd.concat([df, extra], ignore_index=True), root, LABELS)
    summary = report.summary()
    assert summary["missing_files"] == 1
    assert summary["unreadable_images"] == 1
    assert summary["invalid_labels"] == 1
    assert report.n_valid == 14
    assert not report.ok
    # X3 reuses an existing file under a new id → byte-identical duplicate
    assert summary["exact_duplicate_groups"] == 1


def test_lesion_level_split_never_shares_lesions(tiny_dataset):
    csv, _ = tiny_dataset
    df = read_metadata_csv(csv)
    out = assign_splits(df, test_fraction=0.5, group_column="lesion_id", seed=1)
    assert out.attrs["split_level"] == "lesion"
    dev = set(out.loc[out.split == "dev", "lesion_id"])
    test = set(out.loc[out.split == "test", "lesion_id"])
    assert dev and test and not dev & test


def test_split_falls_back_to_image_level_without_ids(tiny_dataset):
    csv, _ = tiny_dataset
    df = read_metadata_csv(csv).assign(lesion_id=pd.NA)
    out = assign_splits(df, 1.0, "lesion_id", seed=0)
    assert out.attrs["split_level"] == "image" and (out.split == "test").all()


def test_subsample_keeps_every_class(tiny_dataset):
    csv, _ = tiny_dataset
    df = read_metadata_csv(csv)
    sub = stratified_group_subsample(df, 7, "lesion_id", seed=0)
    assert set(sub["label"]) == set(LABELS)


def test_prepare_dataset(tmp_config):
    prepared = prepare_dataset(tmp_config)
    assert len(prepared.table) == 14
    assert prepared.split_level == "lesion"


def test_save_image_never_overwrites(tmp_path, sample_image):
    path = save_image(sample_image, tmp_path / "a")
    with pytest.raises(FileExistsError):
        save_image(sample_image, tmp_path / "a")
    assert np.array_equal(load_image(path), sample_image)  # PNG is lossless


def test_isic_groundtruth_conversion(tmp_path):
    csv = tmp_path / "gt.csv"
    pd.DataFrame({"image": ["ISIC_1", "ISIC_2"], "MEL": [1.0, 0], "NV": [0, 1.0], "BCC": 0, "AKIEC": 0,
                  "BKL": 0, "DF": 0, "VASC": 0}).to_csv(csv, index=False)
    out = convert_isic2018_groundtruth(csv)
    assert out["label"].tolist() == ["mel", "nv"]
    assert out["image_path"].tolist() == ["ISIC_1.jpg", "ISIC_2.jpg"]


def test_zip_slip_is_rejected(tmp_path):
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_lesion_lookup_survives_dropped_connections(tmp_path, monkeypatch):
    import http.client
    import io
    import json as _json

    from skin_research.data import metadata as md

    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if "BAD" in req.full_url:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        body = _json.dumps({"metadata": {"clinical": {"lesion_id": "IL_1"}}}).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(md.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(md.time, "sleep", lambda s: None)
    cache = tmp_path / "cache.csv"
    out = md.fetch_isic_lesion_ids(["ISIC_OK", "ISIC_BAD"], cache, delay_s=0)
    assert out.set_index("image_id").loc["ISIC_OK", "lesion_id"] == "IL_1"
    assert pd.isna(out.set_index("image_id").loc["ISIC_BAD", "lesion_id"])
    cached = pd.read_csv(cache)
    assert list(cached["image_id"]) == ["ISIC_OK"]  # failures are not cached, so a rerun retries them
