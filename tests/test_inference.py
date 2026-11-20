"""Inference tests.

Unit tests use a tiny deterministic stub adapter purely to check label
re-ordering and table construction. Tests marked ``model`` use the real
pretrained checkpoint and are *skipped* (not passed) if it is not cached.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from skin_research.config import load_config
from skin_research.inference.model_loader import (
    ModelAdapter,
    ModelLoadError,
    cached_model_files,
    check_label_compatibility,
    load_model,
)
from skin_research.inference.predictor import Predictor

from .conftest import LABELS, synthetic_lesion

MODEL_LABELS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
LABEL_MAP = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AKIEC": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc"}


class StubAdapter(ModelAdapter):
    """Test double: logit k is large for model class k = (mean pixel // 10) % 7."""

    name = "stub"

    def __init__(self):  # noqa: D107 - no config needed
        self.device = torch.device("cpu")

    @property
    def labels(self):
        return MODEL_LABELS

    def load(self, local_only=False):
        pass

    def preprocess(self, images):
        return torch.tensor([[float(img.mean())] for img in images])

    def forward(self, batch):
        idx = (batch[:, 0] // 10).long() % 7
        return torch.nn.functional.one_hot(idx, 7).float() * 10


def test_label_compatibility_checks():
    assert check_label_compatibility(MODEL_LABELS, LABEL_MAP, LABELS).ok
    bad = check_label_compatibility(MODEL_LABELS + ["SCC"], LABEL_MAP, LABELS)
    assert not bad.ok and "SCC" in bad.problems[0]
    missing = check_label_compatibility(MODEL_LABELS[:-1], LABEL_MAP, LABELS)
    assert not missing.ok and "vasc" in " ".join(missing.problems)


def test_predictor_reorders_probabilities_to_dataset_labels():
    predictor = Predictor(StubAdapter(), LABEL_MAP, LABELS, batch_size=2)
    img = np.full((8, 8, 3), 5, np.uint8)  # mean 5 → model class 0 = MEL
    probs = predictor.predict_arrays([img, img, img])
    assert probs.shape == (3, 7)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert LABELS[probs[0].argmax()] == "mel"


def test_predict_stream_builds_prediction_table():
    predictor = Predictor(StubAdapter(), LABEL_MAP, LABELS, batch_size=2)
    items = [({"image_id": f"i{i}", "label": "mel"}, np.full((8, 8, 3), 5 + 10 * i, np.uint8)) for i in range(3)]
    df = pd.concat(list(predictor.predict_stream(items)), ignore_index=True)
    assert len(df) == 3 and predictor.images_seen == 3
    assert df["predicted_label"].tolist() == ["mel", "nv", "bcc"]
    assert df["correct"].tolist() == [True, False, False]
    assert {"confidence", "margin", "entropy"} <= set(df.columns)


def test_unknown_adapter_is_a_clear_error(tmp_config):
    tmp_config.model.adapter = "nonexistent"
    with pytest.raises(ModelLoadError, match="Unknown model adapter"):
        load_model(tmp_config)


def test_missing_weights_give_clear_error(tmp_config):
    """Offline + empty cache must raise, never fall back to fake predictions."""
    with pytest.raises(ModelLoadError, match="No predictions will be produced"):
        load_model(tmp_config, local_only=True)  # tmp_config points models_dir at an empty folder


real_model = pytest.mark.skipif(not cached_model_files(load_config()),
                                reason="pretrained weights not cached; run `skin-research download-model`")


@pytest.mark.model
@real_model
def test_real_model_loads_and_predicts():
    cfg = load_config()
    adapter = load_model(cfg, local_only=True)
    assert adapter.labels == MODEL_LABELS
    meta = adapter.metadata()
    assert meta["parameters"] == 4_016_515 and meta["input_size"] == 320
    predictor = Predictor(adapter, cfg.model.label_map, cfg.dataset.labels, batch_size=2)
    probs = predictor.predict_arrays([synthetic_lesion(i, (450, 600)) for i in range(3)])
    assert probs.shape == (3, 7)
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
    # identical input → identical output (deterministic eval mode, dropout off)
    again = predictor.predict_arrays([synthetic_lesion(0, (450, 600))])
    np.testing.assert_allclose(probs[0], again[0], atol=1e-6)
