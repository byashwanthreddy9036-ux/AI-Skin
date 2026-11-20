"""Controlled, parameterised image degradations simulating poor acquisition.

Every function takes an RGB ``uint8`` array and returns a *new* array of the same
shape (inputs are never modified). Stochastic degradations draw from the
``rng`` passed in, so results are reproducible from a seed.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from PIL import Image

Degradation = Callable[..., np.ndarray]


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, rng: np.random.Generator | None = None, *, sigma: float) -> np.ndarray:
    """Defocus-like blur: Gaussian kernel with standard deviation ``sigma`` pixels."""
    if sigma <= 0:
        return image.copy()
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def motion_kernel(kernel_size: int, angle: float) -> np.ndarray:
    """Normalised linear motion-blur kernel of length ``kernel_size`` at ``angle`` degrees."""
    k = max(1, int(kernel_size))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k), flags=cv2.INTER_LINEAR)
    return kernel / kernel.sum()


def motion_blur(image: np.ndarray, rng: np.random.Generator | None = None, *,
                kernel_size: int, angle: float = 0.0) -> np.ndarray:
    """Camera/hand-shake blur along a straight line."""
    if kernel_size <= 1:
        return image.copy()
    return cv2.filter2D(image, -1, motion_kernel(kernel_size, angle), borderType=cv2.BORDER_REFLECT)


def underexposure(image: np.ndarray, rng: np.random.Generator | None = None, *,
                  gamma: float, gain: float = 1.0) -> np.ndarray:
    """Poor lighting: gamma > 1 darkens mid-tones, gain < 1 scales overall exposure."""
    x = image.astype(np.float32) / 255.0
    return _to_uint8(gain * np.power(x, gamma) * 255.0)


def overexposure(image: np.ndarray, rng: np.random.Generator | None = None, *, gain: float) -> np.ndarray:
    """Too much light: linear gain with highlight clipping (information is lost)."""
    return _to_uint8(image.astype(np.float32) * gain)


def low_contrast(image: np.ndarray, rng: np.random.Generator | None = None, *, factor: float) -> np.ndarray:
    """Compress intensities towards the image's mean grey level by ``factor`` (0-1)."""
    x = image.astype(np.float32)
    mean = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean()
    return _to_uint8(mean + (x - mean) * factor)


def low_resolution(image: np.ndarray, rng: np.random.Generator | None = None, *, scale: float) -> np.ndarray:
    """Capture at lower resolution: downsample by ``scale``, then upsample back.

    Returning the original size keeps the model's preprocessing identical across
    conditions; the lost detail cannot be recovered by the upsampling.
    """
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def jpeg_compression(image: np.ndarray, rng: np.random.Generator | None = None, *, quality: int) -> np.ndarray:
    """Lossy JPEG round trip at the given quality (1-95): blocking and ringing artefacts."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def gaussian_noise(image: np.ndarray, rng: np.random.Generator | None = None, *, sigma: float) -> np.ndarray:
    """Additive Gaussian sensor noise with standard deviation ``sigma`` grey levels."""
    rng = rng or np.random.default_rng(0)
    return _to_uint8(image.astype(np.float32) + rng.normal(0.0, sigma, image.shape).astype(np.float32))


DEGRADATIONS: dict[str, Degradation] = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "underexposure": underexposure,
    "overexposure": overexposure,
    "low_contrast": low_contrast,
    "low_resolution": low_resolution,
    "jpeg_compression": jpeg_compression,
    "gaussian_noise": gaussian_noise,
}


def apply_degradation(image: np.ndarray, name: str, params: dict[str, Any], seed: int | None = None) -> np.ndarray:
    """Apply a registered degradation with explicit parameters."""
    if name not in DEGRADATIONS:
        raise ValueError(f"Unknown degradation '{name}'. Available: {', '.join(DEGRADATIONS)}")
    try:
        return DEGRADATIONS[name](image, np.random.default_rng(seed), **params)
    except TypeError as exc:
        raise ValueError(f"Bad parameters {params} for degradation '{name}': {exc}") from exc
