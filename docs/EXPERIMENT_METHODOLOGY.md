# Experiment methodology

## Hypotheses

- **H1 (degradation):** controlled reductions in image quality (blur, poor lighting, overexposure, low
  contrast, low resolution, compression, noise) reduce classification performance, and the reduction grows
  with severity.
- **H2 (enhancement of good images):** classical enhancement of already good-quality dermoscopic images does
  *not* reliably improve performance, because the model was trained on similar unenhanced images (a
  distribution shift in either direction can hurt).
- **H3 (restoration):** enhancement matched to a specific degradation (e.g. brightness correction for
  underexposure) recovers part of the performance lost to that degradation.
- **H4 (reliability):** degradation also harms reliability: predictions flip, calibration worsens, and wrong
  predictions are still made with substantial confidence.

The null hypothesis for every comparison is "no difference from the reference condition".

## Design

A **within-image (paired) design**: every condition is applied to the same evaluation images, and the
model, weights and preprocessing are identical. Any difference between conditions is therefore caused by
the transformation (plus sampling noise), not by different images or models.

## Experimental conditions

| Code (brief) | Condition(s) in this system |
|---|---|
| A. Original | `original` |
| B. Gaussian blur | `gaussian_blur__{mild,moderate,severe}` (σ = 1.5 / 3 / 6 px) |
| C. Poor lighting | `underexposure__*` (gamma 1.5/2.0/2.8 with gain 0.85/0.65/0.45) |
| D. Overexposure | `overexposure__*` (gain 1.25/1.6/2.1, highlights clip) |
| E. Low resolution | `low_resolution__*` (scale 1/2, 1/4, 1/8 then back to original size) |
| F. Low contrast | `low_contrast__*` (factor 0.7/0.45/0.25 towards mean grey) |
| G. Sharpening | `enhance__sharpen` (unsharp mask σ 1.5, amount 1) |
| H. CLAHE | `enhance__clahe` (clip 2.0, 8×8 tiles, lightness only) |
| I. Enhancement after degradation | 11 pairs, e.g. `underexposure__moderate+brightness_correction`, `gaussian_blur__moderate+deblur`, `gaussian_noise__moderate+denoise` |
| extra | `motion_blur__*`, `jpeg_compression__*`, `gaussian_noise__*`, `enhance__{brightness_correction,contrast_stretch,denoise}` |

All parameters are in `configs/config.yaml`. They were **fixed before** looking at results and were not
tuned on the evaluation set. Images are 600×450 and the model resizes to 320×320, so a σ = 3 px blur at
full resolution is ≈1.6 px at model resolution.

## Dataset splitting

The model is only evaluated, never trained, so all valid images form the evaluation split by default
(`test_fraction: 1.0`). If a dev split is needed (e.g. to tune enhancement parameters), set
`test_fraction < 1`: `StratifiedGroupKFold` keeps every lesion (or patient, when available) in one split
only. If no complete lesion/patient column exists, the system falls back to image-level splitting and logs a
warning that leakage cannot be excluded. The split is written to `data/processed/<experiment>_split.csv`
**before** any variation is generated.

## Label handling under degradation

Ground-truth labels are kept for every transformed image (the lesion has not changed). Each variant is also
quality-assessed; heuristic flags (`blurry`, `too_dark`, `too_bright`, `low_contrast`, `clipped`,
`low_resolution`) mark images whose diagnostic usefulness is *questionable*. The report gives the
questionable fraction per condition and accuracy separately for flagged and unflagged images, so severe
conditions can be interpreted with that caveat.

## Evaluation methodology

- **Metrics:** accuracy, balanced accuracy (primary; robust to the 64% nevus majority), macro precision,
  recall, F1 and specificity, per-class sensitivity/specificity/F1, one-vs-rest ROC-AUC (only for classes
  with positives and negatives), confusion matrix.
- **Reliability:** mean top-1 confidence, confidence when correct vs wrong, expected calibration error
  (10 bins), Brier score, negative log-likelihood, prediction **flip rate** vs original, mean change in the
  probability of the true class.
- **Macro averaging** uses classes present in the ground truth (a class with no true examples has undefined
  recall and is excluded rather than counted as 0).
- **Confidence intervals:** 1,000-resample **lesion-clustered bootstrap**, 95% percentile intervals. Images
  of the same lesion are correlated; resampling lesions avoids over-narrow intervals.
- **Paired comparisons:** the difference between a condition and the original uses the *same* resampled
  lesions for both, giving a CI for the difference. **McNemar's exact test** on per-image correctness is
  reported with **Holm–Bonferroni** correction across all comparisons. (McNemar treats images as
  independent; with repeated lesions it is slightly anti-conservative, which is one reason the clustered
  bootstrap CI is the primary criterion.)
- **Restoration:** each degrade+enhance condition is compared with its degraded input; the *recovery
  fraction* = (enhanced − degraded) / (original − degraded) on balanced accuracy.
- **Decision rule in the report:** an effect is only described as a decrease/increase when the 95% CI of
  the paired difference excludes 0. "Not distinguishable" is not evidence of no effect, particularly with
  only 193 images.

## Reproducibility

- Global seed (`seed: 42`) for Python, NumPy and PyTorch; per-image noise seeds derive from
  SHA-256(seed, image_id, condition), so results do not depend on processing order or subsetting.
- Pinned model revision and recorded weights hash; pinned package versions in `uv.lock`.
- Every run stores `config_used.yaml`, `run_info.json` (environment, package versions, git commit, timings),
  the split CSV and the variation manifest.
- Generated images are deterministic; existing outputs are reused only if their recipe (`condition.json`)
  matches exactly.
- Model inference runs in `eval()` mode (dropout disabled) under `torch.inference_mode()`; identical inputs
  give identical outputs (tested).
