"""Batched CPU inference producing tidy prediction tables."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skin_research.data.loader import load_image
from skin_research.inference.model_loader import ModelAdapter
from skin_research.utils.logging import get_logger

log = get_logger(__name__)


class Predictor:
    """Run a model adapter over images in fixed-size batches.

    Probabilities are re-ordered from the model's label order into the dataset's
    canonical label order using ``label_map``, so downstream code never needs
    to know the model's internal class indices.
    """

    def __init__(self, adapter: ModelAdapter, label_map: dict[str, str], dataset_labels: list[str],
                 batch_size: int = 16):
        self.adapter = adapter
        self.batch_size = max(1, batch_size)
        self.dataset_labels = list(dataset_labels)
        model_labels = adapter.labels
        mapped = [label_map.get(m) for m in model_labels]
        # column index in model output for each dataset label (None if the model cannot predict it)
        self._order = [mapped.index(lbl) if lbl in mapped else None for lbl in self.dataset_labels]
        self.images_seen = 0
        self.seconds = 0.0

    def predict_arrays(self, images: list[np.ndarray]) -> np.ndarray:
        """Probabilities, shape (N, len(dataset_labels)), for in-memory RGB images."""
        out = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            t0 = time.perf_counter()
            probs = self.adapter.predict_proba(batch)
            self.seconds += time.perf_counter() - t0
            self.images_seen += len(batch)
            out.append(np.stack([probs[:, j] if j is not None else np.zeros(len(batch)) for j in self._order], axis=1))
        return np.concatenate(out) if out else np.zeros((0, len(self.dataset_labels)))

    def predict_stream(self, items: Iterable[tuple[dict[str, Any], np.ndarray]]) -> Iterator[pd.DataFrame]:
        """Consume ``(record, image)`` pairs lazily and yield prediction frames per batch."""
        records: list[dict[str, Any]] = []
        images: list[np.ndarray] = []
        for record, image in items:
            records.append(record)
            images.append(image)
            if len(images) == self.batch_size:
                yield self._frame(records, self.predict_arrays(images))
                records, images = [], []
        if images:
            yield self._frame(records, self.predict_arrays(images))

    def predict_paths(self, paths: list[Path], records: list[dict[str, Any]] | None = None) -> pd.DataFrame:
        """Predict image files (loaded lazily, one batch at a time)."""
        records = records or [{"path": str(p)} for p in paths]
        stream = ((rec, load_image(Path(p))) for rec, p in zip(records, paths))
        frames = list(self.predict_stream(stream))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _frame(self, records: list[dict[str, Any]], probs: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(records)
        for j, label in enumerate(self.dataset_labels):
            df[f"prob_{label}"] = probs[:, j]
        top = probs.argmax(axis=1)
        df["predicted_label"] = [self.dataset_labels[i] for i in top]
        df["confidence"] = probs.max(axis=1)
        sorted_p = np.sort(probs, axis=1)
        df["margin"] = sorted_p[:, -1] - sorted_p[:, -2] if probs.shape[1] > 1 else 1.0
        df["entropy"] = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
        if "label" in df.columns:
            df["correct"] = df["predicted_label"] == df["label"]
        return df

    @property
    def throughput(self) -> float:
        return self.images_seen / self.seconds if self.seconds else 0.0
