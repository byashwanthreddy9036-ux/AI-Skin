"""Pretrained model adapters.

The rest of the system only talks to :class:`ModelAdapter`, so integrating a
different pretrained classifier means writing one subclass and registering it
in :data:`ADAPTERS`. No adapter may fabricate outputs: if weights cannot be
obtained, loading fails with :class:`ModelLoadError`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from skin_research.config import AppConfig, ModelConfig
from skin_research.utils.logging import get_logger
from skin_research.utils.reproducibility import file_sha256

log = get_logger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when model weights or configuration cannot be obtained or loaded."""


@dataclass
class LabelCompatibility:
    ok: bool
    problems: list[str]


def check_label_compatibility(model_labels: list[str], label_map: dict[str, str],
                              dataset_labels: list[str]) -> LabelCompatibility:
    """Metrics are only meaningful if model and dataset classes map one-to-one."""
    problems = []
    unmapped = [m for m in model_labels if m not in label_map]
    if unmapped:
        problems.append(f"model labels without a mapping: {unmapped}")
    mapped = [label_map[m] for m in model_labels if m in label_map]
    outside = sorted(set(mapped) - set(dataset_labels))
    if outside:
        problems.append(f"model labels map to classes absent from the dataset label set: {outside}")
    missing = sorted(set(dataset_labels) - set(mapped))
    if missing:
        problems.append(f"dataset labels the model cannot predict: {missing}")
    if len(set(mapped)) != len(mapped):
        problems.append("several model labels map to the same dataset label")
    return LabelCompatibility(not problems, problems)


class ModelAdapter(ABC):
    """Interface every pretrained classifier must implement."""

    name: str

    def __init__(self, model_cfg: ModelConfig, cache_dir: Path):
        self.cfg = model_cfg
        self.cache_dir = cache_dir
        self.device = torch.device(model_cfg.device)

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Model output labels in logit order."""

    @abstractmethod
    def load(self, local_only: bool = False) -> None:
        """Download (if needed) and load weights. Must raise ModelLoadError on failure."""

    @abstractmethod
    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """Convert RGB uint8 arrays into the model's input batch."""

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (N, len(labels))."""

    def predict_proba(self, images: list[np.ndarray]) -> np.ndarray:
        with torch.inference_mode():
            logits = self.forward(self.preprocess(images).to(self.device))
            return torch.softmax(logits.float(), dim=1).cpu().numpy()

    def metadata(self) -> dict[str, Any]:
        return {"adapter": self.name, "repo_id": self.cfg.repo_id, "revision": self.cfg.revision,
                "labels": self.labels, "device": str(self.device)}


# --------------------------------------------------------------------------- EfficientNet (ISIC 2018)
class _EfficientNetB0Classifier(nn.Module):
    """Architecture published with adeIaide/skin-lesion-7class-efficientnet
    (``inference_example.py``): torchvision EfficientNet-B0 features + pooling,
    dropout and a linear head. Parameter names must match the checkpoint."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        from torchvision import models

        network = models.efficientnet_b0(weights=None)  # weights come from the checkpoint
        embedding_dim = network.classifier[1].in_features
        self.encoder = nn.Sequential(network.features, network.avgpool, nn.Flatten(1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encoder(image)))


class EfficientNetISIC2018Adapter(ModelAdapter):
    name = "efficientnet_isic2018"

    def __init__(self, model_cfg: ModelConfig, cache_dir: Path):
        super().__init__(model_cfg, cache_dir)
        self.model: nn.Module | None = None
        self.hub_config: dict[str, Any] = {}
        self.weights_path: Path | None = None
        self._transform = None

    @property
    def labels(self) -> list[str]:
        if not self.hub_config:
            raise ModelLoadError("Model not loaded yet: call load() first.")
        return list(self.hub_config["class_names"])

    def _download(self, filename: str, local_only: bool = False) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        try:
            return Path(hf_hub_download(self.cfg.repo_id, filename, revision=self.cfg.revision,
                                        cache_dir=self.cache_dir, local_files_only=local_only))
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError, ValueError) as exc:
            raise ModelLoadError(
                f"Could not obtain '{filename}' from Hugging Face repo '{self.cfg.repo_id}' "
                f"(revision {self.cfg.revision}).\nCause: {exc}\n"
                "Check your internet connection, run `skin-research download-model`, or set "
                "HF_HUB_OFFLINE=1 only after the files are cached in ./models. "
                "No predictions will be produced without the real weights."
            ) from exc

    def load(self, local_only: bool = False) -> None:
        from safetensors.torch import load_file
        from torchvision import transforms

        config_path = self._download(self.cfg.config_file, local_only)
        self.weights_path = self._download(self.cfg.weights_file, local_only)
        self.hub_config = json.loads(config_path.read_text(encoding="utf-8"))
        names = self.hub_config.get("class_names")
        if not names or len(names) != int(self.hub_config.get("num_classes", -1)):
            raise ModelLoadError(f"Model config.json has an inconsistent class list: {names}")
        index = self.hub_config.get("class_to_index", {})
        if index and any(index[n] != i for i, n in enumerate(names)):
            raise ModelLoadError("class_names order disagrees with class_to_index in config.json")

        model = _EfficientNetB0Classifier(len(names), float(self.hub_config.get("dropout", 0.3)))
        try:
            state = load_file(str(self.weights_path), device="cpu")
            model.load_state_dict(state, strict=True)  # every tensor must match exactly
        except (RuntimeError, OSError) as exc:
            raise ModelLoadError(f"Weights in {self.weights_path} do not fit the architecture: {exc}") from exc
        self.model = model.to(self.device).eval()

        size = int(self.hub_config["image_size"])
        norm = self.hub_config["normalization"]
        # Identical to the model card's inference_example.py preprocessing.
        self._transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ])
        log.info("Loaded %s (%d classes, input %dx%d) on %s", self.cfg.repo_id, len(names), size, size, self.device)

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        if self._transform is None:
            raise ModelLoadError("Model not loaded yet: call load() first.")
        return torch.stack([self._transform(Image.fromarray(img)) for img in images])

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(batch)

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model else None
        meta.update(
            architecture=self.hub_config.get("architecture"),
            input_size=self.hub_config.get("image_size"),
            normalization=self.hub_config.get("normalization"),
            training=self.hub_config.get("training"),
            reported_evaluation=self.hub_config.get("evaluation"),
            parameters=n_params,
            weights_file=self.cfg.weights_file,
            weights_sha256=file_sha256(self.weights_path) if self.weights_path else None,
            license="CC-BY-NC-4.0",
            source_url=f"https://huggingface.co/{self.cfg.repo_id}",
        )
        return meta


ADAPTERS: dict[str, type[ModelAdapter]] = {
    EfficientNetISIC2018Adapter.name: EfficientNetISIC2018Adapter,
}


def load_model(cfg: AppConfig, local_only: bool = False) -> ModelAdapter:
    """Instantiate and load the adapter named in ``model.adapter``."""
    name = cfg.model.adapter
    if name not in ADAPTERS:
        raise ModelLoadError(f"Unknown model adapter '{name}'. Registered: {', '.join(ADAPTERS)}")
    torch.set_num_threads(max(1, cfg.model.num_threads))
    if cfg.model.device != "cpu" and not torch.cuda.is_available():
        log.warning("model.device=%s requested but CUDA is unavailable; using CPU", cfg.model.device)
        cfg.model.device = "cpu"
    adapter = ADAPTERS[name](cfg.model, cfg.paths.models_dir)
    adapter.load(local_only=local_only)
    return adapter


def cached_model_files(cfg: AppConfig) -> bool:
    """True if the configured weights are already in the local cache (no network)."""
    from huggingface_hub import try_to_load_from_cache

    for filename in (cfg.model.config_file, cfg.model.weights_file):
        hit = try_to_load_from_cache(cfg.model.repo_id, filename, cache_dir=cfg.paths.models_dir,
                                     revision=cfg.model.revision)
        if not isinstance(hit, str):
            return False
    return True
