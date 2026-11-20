import numpy as np
import pytest

from skin_research.config import load_config
from skin_research.preprocessing.degradation import DEGRADATIONS, apply_degradation, motion_kernel
from skin_research.quality import metrics as m


@pytest.mark.parametrize("name", list(DEGRADATIONS))
def test_every_configured_severity_runs_and_preserves_input(sample_image, name):
    original = sample_image.copy()
    for severity, params in load_config().degradations[name].items():
        out = apply_degradation(sample_image, name, params, seed=1)
        assert out.shape == sample_image.shape and out.dtype == np.uint8, severity
        assert not np.array_equal(out, sample_image), f"{name}/{severity} changed nothing"
    assert np.array_equal(sample_image, original), "input array was modified"


def test_blur_severity_is_monotonic(sample_image):
    params = load_config().degradations["gaussian_blur"]
    scores = [m.laplacian_variance(m.to_gray(apply_degradation(sample_image, "gaussian_blur", params[s])))
              for s in ("mild", "moderate", "severe")]
    assert scores[0] > scores[1] > scores[2]


def test_exposure_direction(sample_image):
    dark = apply_degradation(sample_image, "underexposure", {"gamma": 2.0, "gain": 0.6})
    bright = apply_degradation(sample_image, "overexposure", {"gain": 1.6})
    assert m.brightness(dark) < m.brightness(sample_image) < m.brightness(bright)


def test_low_contrast_and_resolution(sample_image):
    flat = apply_degradation(sample_image, "low_contrast", {"factor": 0.3})
    assert m.rms_contrast(flat) < 0.5 * m.rms_contrast(sample_image)
    low = apply_degradation(sample_image, "low_resolution", {"scale": 0.25})
    assert low.shape == sample_image.shape
    assert m.laplacian_variance(m.to_gray(low)) < m.laplacian_variance(m.to_gray(sample_image))


def test_noise_is_reproducible_from_seed(sample_image):
    a = apply_degradation(sample_image, "gaussian_noise", {"sigma": 20}, seed=7)
    b = apply_degradation(sample_image, "gaussian_noise", {"sigma": 20}, seed=7)
    c = apply_degradation(sample_image, "gaussian_noise", {"sigma": 20}, seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_motion_kernel_is_normalised():
    k = motion_kernel(15, 45)
    assert k.shape == (15, 15) and abs(k.sum() - 1) < 1e-5


def test_unknown_degradation_and_bad_params(sample_image):
    with pytest.raises(ValueError):
        apply_degradation(sample_image, "fog", {})
    with pytest.raises(ValueError):
        apply_degradation(sample_image, "gaussian_blur", {"radius": 3})
