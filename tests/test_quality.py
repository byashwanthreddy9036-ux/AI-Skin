import cv2
import numpy as np
import pytest

from skin_research.quality import metrics as m
from skin_research.quality.assessment import assess_array, assess_file


def test_blur_lowers_laplacian_and_tenengrad(sample_image):
    gray = m.to_gray(sample_image)
    scores = [m.laplacian_variance(cv2.GaussianBlur(gray, (0, 0), s)) for s in (0.5, 2, 5)]
    assert m.laplacian_variance(gray) > scores[0] > scores[1] > scores[2]
    assert m.tenengrad(gray) > m.tenengrad(cv2.GaussianBlur(gray, (0, 0), 3))


def test_brightness_and_contrast_respond_to_changes(sample_image):
    darker = (sample_image * 0.5).astype(np.uint8)
    flat = (128 + (sample_image.astype(np.float32) - 128) * 0.3).astype(np.uint8)
    assert m.brightness(darker) < m.brightness(sample_image)
    assert m.rms_contrast(flat) < m.rms_contrast(sample_image)
    assert 0 <= m.brightness(sample_image) <= 1


def test_uniform_image_edge_cases():
    black = np.zeros((64, 64, 3), np.uint8)
    white = np.full((64, 64, 3), 255, np.uint8)
    assert m.laplacian_variance(m.to_gray(black)) == 0
    assert m.rms_contrast(white) == 0
    assert m.clipped_fractions(black) == (1.0, 0.0)
    assert m.clipped_fractions(white) == (0.0, 1.0)


def test_noise_estimate_increases_with_noise(sample_image):
    rng = np.random.default_rng(0)
    noisy = np.clip(sample_image + rng.normal(0, 25, sample_image.shape), 0, 255).astype(np.uint8)
    assert m.noise_sigma(m.to_gray(noisy)) > m.noise_sigma(m.to_gray(sample_image))


def test_flags_raised_for_dark_and_flat_images():
    dark = np.full((100, 100, 3), 10, np.uint8)
    record = assess_array(dark)
    assert "too_dark" in record["quality_flags"] and "low_contrast" in record["quality_flags"]
    assert record["label_usefulness"] == "questionable"


def test_metrics_reject_wrong_input():
    with pytest.raises(ValueError):
        m.compute_metrics(np.zeros((10, 10), np.uint8))


def test_assess_file_handles_corrupt_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")
    record = assess_file(bad)
    assert record["valid"] is False and record["error"]
