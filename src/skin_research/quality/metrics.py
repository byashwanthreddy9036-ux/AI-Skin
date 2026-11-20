"""Low-level, no-reference image quality metrics.

All functions take an RGB ``uint8`` array. These are classical heuristics:
each captures one aspect of quality, none is a universally reliable judgement.
Blur and sharpness scores in particular depend on resolution and image content
(a smooth lesion on smooth skin legitimately has few edges), so they are only
comparable between images of similar content at the same analysis size.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    """Resize so the longest side equals ``long_side`` (area interpolation when shrinking)."""
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1) < 1e-6:
        return image
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)


def brightness(image: np.ndarray) -> float:
    """Mean perceptual lightness (CIELAB L*), scaled to [0, 1]."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # OpenCV stores L* in 0..255 for uint8
    return float(lab[..., 0].mean() / 255.0)


def rms_contrast(image: np.ndarray) -> float:
    """Root-mean-square contrast: standard deviation of grey levels in [0, 1]."""
    return float(to_gray(image).astype(np.float32).std() / 255.0)


def percentile_contrast(image: np.ndarray, low: float = 5, high: float = 95) -> float:
    """Spread between the low and high grey-level percentiles, in [0, 1] (robust to outliers)."""
    p_low, p_high = np.percentile(to_gray(image), [low, high])
    return float((p_high - p_low) / 255.0)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the classic focus/blur measure (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())


def tenengrad(gray: np.ndarray) -> float:
    """Tenengrad sharpness: mean squared Sobel gradient magnitude (higher = sharper)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def clipped_fractions(image: np.ndarray, dark: int = 5, bright: int = 250) -> tuple[float, float]:
    """Fractions of pixels crushed to black / blown to white (exposure clipping)."""
    gray = to_gray(image)
    return float((gray <= dark).mean()), float((gray >= bright).mean())


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkær (1996) fast noise standard-deviation estimate, in grey levels.

    Uses a Laplacian-difference kernel that cancels smooth image structure;
    strong textures inflate the estimate, so treat it as relative.
    """
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = gray.shape
    response = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (w - 2) * (h - 2)))


def colorfulness(image: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric."""
    r, g, b = (image[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def compute_metrics(image: np.ndarray, analysis_long_side: int = 512) -> dict[str, float]:
    """All pixel-level metrics for one image.

    Brightness/contrast/clipping/colourfulness use the full image; blur,
    sharpness and noise use a copy resized to ``analysis_long_side`` so that
    scores are comparable across image resolutions.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image of shape (H, W, 3).")
    analysed = to_gray(resize_long_side(image, analysis_long_side))
    dark, bright = clipped_fractions(image)
    return {
        "brightness": brightness(image),
        "rms_contrast": rms_contrast(image),
        "percentile_contrast": percentile_contrast(image),
        "laplacian_var": laplacian_variance(analysed),
        "tenengrad": tenengrad(analysed),
        "noise_sigma": noise_sigma(analysed),
        "dark_clipped": dark,
        "bright_clipped": bright,
        "colorfulness": colorfulness(image),
    }
