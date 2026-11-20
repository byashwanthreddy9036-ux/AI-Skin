"""Seeding and run-environment capture for reproducible experiments."""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_TRACKED_PACKAGES = [
    "torch", "torchvision", "numpy", "pandas", "pillow", "opencv-python-headless",
    "scikit-learn", "scipy", "matplotlib", "safetensors", "huggingface-hub",
]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def derived_seed(base_seed: int, *parts: object) -> int:
    """Deterministic per-item seed, independent of processing order.

    Noise added to image ``X`` under condition ``Y`` is always identical, no
    matter how many images are processed or in which order.
    """
    text = "|".join([str(base_seed), *map(str, parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large files never sit fully in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info(root: Path) -> dict[str, Any]:
    """Collect versions and platform details for the experiment record."""
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.machine(),
        "git_commit": git_commit(root),
        "packages": packages,
    }
