# Architecture

## Research question

> Does improving the quality of skin lesion images improve the classification
> accuracy and reliability of an existing pretrained AI model?

The system holds the **model fixed** and varies only the **input image quality**:
original → controlled degradation → enhancement. Any change in performance can
then be attributed to the image transformation (within the limits described in
`EXPERIMENT_METHODOLOGY.md`).

## Phase 0 environment findings (2026-11-20)

| Item | Finding |
|---|---|
| OS | Ubuntu 24.04.3 LTS on WSL2 (kernel 6.6) |
| CPU | 6 logical cores |
| RAM | ~10 GiB visible to WSL (≈8 GiB available) |
| GPU | None (`nvidia-smi` absent) → CPU-only inference |
| Python | System 3.12.3; project uses 3.11 installed via `uv` |
| Package manager | `uv` 0.12 (isolated `.venv`) |
| Git | 2.43, SSH access to GitHub |
| Disk | > 800 GB free |

## Model selection

Candidates investigated on the Hugging Face Hub (model card, config, weights,
licence, label map were checked for each):

| Model | Arch / size | Labels | Training data & split | Licence | Verdict |
|---|---|---|---|---|---|
| `adeIaide/skin-lesion-7class-efficientnet` | EfficientNet-B0, 16 MB | 7 ISIC-2018 classes | ISIC 2018 Task 3 train (= HAM10000), **lesion-disjoint** dev split, macro-F1 0.764 | CC-BY-NC-4.0 | **Selected** |
| `Anwarkh1/Skin_Cancer-Image_Classification` | ViT-B/16, 343 MB | 7 HAM10000 classes | `marmal88/skin_cancer` (HAM10000); split method undocumented, val acc 0.97 suggests lesion leakage | Apache-2.0 | Alternative |
| `gianlab/swin-tiny-...-skin-cancer` | Swin-T, 110 MB | 7 HAM10000 classes | HAM10000, 1 epoch, acc 0.73 | Apache-2.0 | Weak |
| `NeuronZero/SkinCancerClassifier` | Swin, 347 MB | 8 classes (adds SCC) | ISIC 2019-style | Apache-2.0 | Label set differs |

Why EfficientNet-B0:

* Smallest verified checkpoint, runs fast on CPU and uses little RAM.
* Architecture code is published (plain `torchvision`), so no `transformers`
  dependency is needed.
* Label order and preprocessing (320×320, ImageNet normalisation) are stated in
  `config.json`.
* Card reports macro-F1/balanced accuracy/AUROC on a lesion-disjoint split, and
  states its limitations honestly.

Caveats: low adoption (new model, Sept 2026) and claims cannot be independently
verified — which is exactly why we measure it ourselves on held-out data.

## Evaluation data and leakage

The model was trained on ISIC 2018 Task 3 **training** data, which *is* HAM10000.
Evaluating on HAM10000 would therefore measure performance on training images.
The default evaluation set is the **ISIC 2018 Task 3 Validation set** (193
images, 53 MB), which was never part of the training release. The Test set
(1,512 images, 420 MB) is supported for larger runs. HAM10000 is supported as a
loader, but the tool warns about leakage when it is used with this model.

## Module layout

```
src/skin_research/
├── cli.py               # argparse CLI: doctor, validate-data, ..., report, demo
├── config.py            # YAML → typed dataclasses; paths resolved from project root
├── experiment.py        # (added) end-to-end runner: split → variations → inference → evaluation
├── pipeline.py          # (added) post-inference stages shared by experiment/evaluate/report
├── data/
│   ├── download.py      # (added) ISIC dataset registry, sizes, confirmation, zip-slip-safe extraction
│   ├── metadata.py      # CSV schema, ISIC/HAM10000 → canonical metadata converters, ISIC API lesion ids
│   ├── loader.py        # lazy dataset iteration, group-aware splitting
│   └── validation.py    # missing files, corrupt images, labels, duplicates
├── quality/
│   ├── metrics.py       # brightness, contrast, Laplacian blur, Tenengrad sharpness
│   └── assessment.py    # per-image quality table + summary
├── preprocessing/
│   ├── degradation.py   # 8 configurable degradations × severity levels
│   ├── enhancement.py   # brightness/contrast correction, CLAHE, sharpening, denoising, RL deblur
│   └── variations.py    # (added) experimental conditions + traceable generation/manifest
├── inference/
│   ├── model_loader.py  # ModelAdapter interface + EfficientNet adapter + registry
│   └── predictor.py     # batched CPU inference → probabilities table
├── evaluation/
│   ├── metrics.py       # acc, P/R/F1, balanced acc, sens/spec, ROC-AUC, CIs
│   └── comparison.py    # condition-vs-baseline deltas, paired bootstrap, McNemar
├── reporting/
│   ├── charts.py        # Matplotlib figures
│   └── report.py        # Markdown research report
└── utils/
    ├── logging.py
    └── reproducibility.py
```

## Data flow

```
 metadata.csv + images (read-only, data/raw)
        │ validate
        ▼
 evaluation split (fixed seed, group-aware)  ──► quality assessment
        │
        ├─► original ───────────────────────────────┐
        ├─► degradations ──► data/variations/…      │
        │        └─► enhancement-after-degradation  ├─► Predictor (one model, one
        └─► enhancement ──► data/variations/…       │   preprocessing pipeline)
                                                    ▼
                         variations manifest (original_id, transform, params, path, label)
                                                    │
                                                    ▼
                         outputs/predictions/*.csv ──► evaluation ──► charts ──► report.md
```

Every generated image is recorded in a manifest with its source image ID,
transformation and parameters, and ground-truth label. Original images are never
written to.

### Deviations from the suggested structure

* The repository root *is* the project (`AI-Skin/` instead of `skin-image-quality-research/`).
* Added `data/download.py`, `preprocessing/variations.py`, `experiment.py` and `pipeline.py` so that
  downloading, condition generation, orchestration and post-processing each have one home instead of
  being mixed into the loader or CLI.
* `docs/` holds the three required documents plus this file.

## Memory and performance

Images are decoded lazily one source image at a time; variants are streamed straight into fixed-size
inference batches. Measured on the target machine (6-core WSL2, CPU only): peak RSS ≈ 0.8 GB, ≈ 9–10
images/s with batch 8 and 3 threads (more threads were slower under WSL2).

## Model adapter

`ModelAdapter` defines `labels`, `preprocess(images)`, `predict_proba(batch)` and
`metadata()`. A registry maps config names to adapters, so a second classifier
(e.g. a Hugging Face `transformers` ViT) can be added by writing one class. The
adapter also declares a `label_map` that maps the model's output labels to the
dataset's canonical labels; metrics are refused if any dataset label cannot be
mapped.
