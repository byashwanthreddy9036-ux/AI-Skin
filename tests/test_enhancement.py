import numpy as np
import pytest

from skin_research.config import load_config
from skin_research.preprocessing.degradation import apply_degradation
from skin_research.preprocessing.enhancement import ENHANCEMENTS, apply_enhancement
from skin_research.quality import metrics as m


@pytest.mark.parametrize("name", list(ENHANCEMENTS))
def test_every_enhancement_runs_and_preserves_input(sample_image, name):
    original = sample_image.copy()
    out = apply_enhancement(sample_image, name, load_config().enhancements[name])
    assert out.shape == sample_image.shape and out.dtype == np.uint8
    assert np.array_equal(sample_image, original)


def test_brightness_correction_recovers_dark_image(sample_image):
    dark = apply_degradation(sample_image, "underexposure", {"gamma": 2.0, "gain": 0.6})
    fixed = apply_enhancement(dark, "brightness_correction", {"target_mean": 0.55, "max_gamma": 3.0})
    assert abs(m.brightness(fixed) - 0.55) < abs(m.brightness(dark) - 0.55)


def test_contrast_enhancements_raise_contrast(sample_image):
    flat = apply_degradation(sample_image, "low_contrast", {"factor": 0.3})
    for name in ("contrast_stretch", "clahe"):
        out = apply_enhancement(flat, name, load_config().enhancements[name])
        assert m.rms_contrast(out) > m.rms_contrast(flat), name


def test_sharpen_and_deblur_increase_sharpness(sample_image):
    blurred = apply_degradation(sample_image, "gaussian_blur", {"sigma": 2.0})
    base = m.tenengrad(m.to_gray(blurred))
    assert m.tenengrad(m.to_gray(apply_enhancement(blurred, "sharpen", {"sigma": 1.5, "amount": 1.0}))) > base
    assert m.tenengrad(m.to_gray(apply_enhancement(blurred, "deblur", {"psf_sigma": 2.0, "iterations": 10}))) > base


def test_denoise_reduces_noise(sample_image):
    noisy = apply_degradation(sample_image, "gaussian_noise", {"sigma": 20}, seed=0)
    clean = apply_enhancement(noisy, "denoise", load_config().enhancements["denoise"])
    assert m.noise_sigma(m.to_gray(clean)) < m.noise_sigma(m.to_gray(noisy))


def test_brightness_correction_preserves_colour(sample_image):
    import cv2

    out = apply_enhancement(sample_image, "brightness_correction", {"target_mean": 0.4})
    ab_in = cv2.cvtColor(sample_image, cv2.COLOR_RGB2LAB)[..., 1:].astype(int)
    ab_out = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 1:].astype(int)
    assert np.abs(ab_in - ab_out).mean() < 3  # only lightness is meant to change


def test_unknown_enhancement(sample_image):
    with pytest.raises(ValueError):
        apply_enhancement(sample_image, "super_resolution_gan", {})
