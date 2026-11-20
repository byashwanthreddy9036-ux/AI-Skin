"""Classical, deterministic image enhancement.

Deliberately limited to non-generative filters: they redistribute or filter the
pixel information already present and cannot hallucinate lesion structures.
They can still *remove* or *distort* diagnostic detail (e.g. denoising erases
fine pigment network, sharpening amplifies noise), so "enhanced" never means
"better" until measured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

Enhancement = Callable[..., np.ndarray]


def _apply_to_lightness(image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run ``fn`` on the CIELAB L channel only, leaving colour (a*, b*) unchanged."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = fn(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def brightness_correction(image: np.ndarray, *, target_mean: float = 0.55, max_gamma: float = 3.0) -> np.ndarray:
    """Gamma-correct lightness so its mean moves to ``target_mean`` (0-1).

    gamma = log(target) / log(current mean), clamped to [1/max_gamma, max_gamma].
    Clipped highlights from overexposure cannot be restored.
    """
    def fn(L: np.ndarray) -> np.ndarray:
        mean = float(np.clip(L.mean() / 255.0, 1e-3, 1 - 1e-3))
        gamma = float(np.clip(np.log(target_mean) / np.log(mean), 1 / max_gamma, max_gamma))
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
        return cv2.LUT(L, lut)

    return _apply_to_lightness(image, fn)


def contrast_stretch(image: np.ndarray, *, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """Linearly map the given lightness percentiles to the full 0-255 range."""
    def fn(L: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(L, [low_percentile, high_percentile])
        if hi - lo < 1:
            return L
        return np.clip((L.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    return _apply_to_lightness(image, fn)


def clahe(image: np.ndarray, *, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation on lightness."""
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return _apply_to_lightness(image, op.apply)


def sharpen(image: np.ndarray, *, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask: image + amount * (image - GaussianBlur(image, sigma))."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def denoise(image: np.ndarray, *, h: float = 7, h_color: float = 7,
            template_window: int = 7, search_window: int = 21) -> np.ndarray:
    """Non-local means denoising (OpenCV). ``h`` controls strength; higher also removes detail."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV's colour NLM expects BGR
    out = cv2.fastNlMeansDenoisingColored(bgr, None, h, h_color, template_window, search_window)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def deblur(image: np.ndarray, *, psf_sigma: float = 3.0, iterations: int = 20) -> np.ndarray:
    """Richardson–Lucy deconvolution assuming a Gaussian point-spread function.

    Non-blind: it only helps when ``psf_sigma`` roughly matches the actual blur.
    Too many iterations amplify noise and create ringing.
    """
    observed = image.astype(np.float32) / 255.0 + 1e-4
    estimate = observed.copy()
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), psf_sigma, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    for _ in range(int(iterations)):
        ratio = observed / (blur(estimate) + 1e-6)
        estimate *= blur(ratio)  # Gaussian PSF is symmetric, so its mirror is itself
    return np.clip(estimate * 255.0, 0, 255).astype(np.uint8)


ENHANCEMENTS: dict[str, Enhancement] = {
    "brightness_correction": brightness_correction,
    "contrast_stretch": contrast_stretch,
    "clahe": clahe,
    "sharpen": sharpen,
    "denoise": denoise,
    "deblur": deblur,
}


def apply_enhancement(image: np.ndarray, name: str, params: dict[str, Any]) -> np.ndarray:
    """Apply a registered enhancement with explicit parameters."""
    if name not in ENHANCEMENTS:
        raise ValueError(f"Unknown enhancement '{name}'. Available: {', '.join(ENHANCEMENTS)}")
    try:
        return ENHANCEMENTS[name](image, **params)
    except TypeError as exc:
        raise ValueError(f"Bad parameters {params} for enhancement '{name}': {exc}") from exc
