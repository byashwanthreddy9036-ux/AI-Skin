# Model and dataset

## Pretrained model

| Property | Value |
|---|---|
| Repository | <https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet> |
| Pinned revision | `b5ebe73c3b89baa90ee4f991afa318385414a973` |
| Weights file | `model.safetensors` (16.3 MB), SHA-256 `fed0c3cb0a15823f4f6a82d4dd2725a0b8e6068a971664e1504403fba5b54562` |
| Architecture | torchvision EfficientNet-B0 features + global average pool + dropout 0.3 + linear head (4,016,515 parameters) |
| Input | RGB, resized to 320×320, ImageNet mean/std normalisation |
| Training data | ISIC 2018 Task 3 training set (= HAM10000), image only (no metadata) |
| Card-reported evaluation | internal lesion-disjoint dev split (2,026 images): macro-F1 0.764, balanced accuracy 0.796, accuracy 0.864, macro AUROC 0.934 |
| Framework | PyTorch (pure torchvision; no `transformers` needed) |
| Licence | **CC-BY-NC-4.0**, non-commercial research and education only |
| Source code | <https://github.com/adeIaide/skin_project> |

### How it was verified

1. The model card, `config.json`, `inference_example.py` and file listing were read from the Hub API.
2. The architecture in `src/skin_research/inference/model_loader.py` copies the card's published class
   definition; weights load with `strict=True`, so any missing or unexpected tensor would raise an error.
3. `class_names` order is checked against `class_to_index` in `config.json` at load time.
4. Preprocessing is identical to the card's `inference_example.py`.
5. Real inference on held-out ISIC 2018 Validation images gives sensible accuracy (see the report), which
   is consistent with a correct label mapping. A wrong mapping would give near-chance results.

### Label mapping

| Model output index | Model label | Dataset label (`dx`) | Diagnosis |
|---|---|---|---|
| 0 | MEL | mel | Melanoma |
| 1 | NV | nv | Melanocytic nevus |
| 2 | BCC | bcc | Basal cell carcinoma |
| 3 | AKIEC | akiec | Actinic keratosis / intraepithelial carcinoma |
| 4 | BKL | bkl | Benign keratosis-like lesion |
| 5 | DF | df | Dermatofibroma |
| 6 | VASC | vasc | Vascular lesion |

The mapping lives in `configs/config.yaml` (`model.label_map`). `check_label_compatibility()` refuses to
compute metrics unless every model label maps to exactly one dataset label and every dataset label can
be predicted.

### Alternatives investigated

| Model | Why not chosen first |
|---|---|
| `Anwarkh1/Skin_Cancer-Image_Classification` (ViT-B/16, 343 MB, Apache-2.0) | 20× larger; split method undocumented and 0.97 val accuracy after 5 epochs suggests same-lesion leakage. A reasonable second adapter. |
| `gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer` (110 MB) | 1 epoch of training, 0.73 accuracy |
| `NeuronZero/SkinCancerClassifier` (Swin, 347 MB) | 8 classes (adds SCC), label set does not match ISIC 2018 |

### Model limitations

- Seven ISIC 2018 classes only; it cannot recognise other conditions or healthy skin.
- Trained on dermoscopic images; not valid for smartphone or clinical photographs.
- New, little-used checkpoint (published September 2026); the card's metrics are self-reported.
- ISIC/HAM10000 under-represents darker skin tones.

### Adding another model

Write a `ModelAdapter` subclass (`labels`, `load`, `preprocess`, `forward`), register it in `ADAPTERS`,
then set `model.adapter`, `repo_id`, `revision` and `label_map` in the config. Nothing else changes.

## Datasets

### ISIC 2018 Task 3: Validation (default)

- 193 dermoscopic images, 600×450 JPEG, 53 MB. Source: official ISIC Challenge bucket
  `https://isic-archive.s3.amazonaws.com/challenges/2018/`.
- **Held out from the model's training data** (released separately from the training set).
- Class counts: nv 123, bkl 22, mel 21, bcc 15, akiec 8, vasc 3, df 1. Too few df/vasc images for
  stable per-class estimates.
- Lesion IDs are not in the challenge files; `prepare-data --fetch-lesion-ids` retrieves them from the
  public ISIC Archive API (`/api/v2/images/<id>/`). 193 images belong to 145 lesions. Patient IDs are not
  public, so **patient-level independence cannot be guaranteed**.

### ISIC 2018 Task 3: Test (optional, recommended for stronger statistics)

- 1,512 images, 420 MB. Also held out. `skin-research download-data --dataset isic2018_test`
  (asks for confirmation), then set `dataset.name`, `metadata_csv` and `image_root` accordingly
  (see README).

### ISIC 2018 Training / HAM10000 (supported, but leaks)

- 10,015 images (2.7 GB) with official lesion groupings. Supported via `--dataset isic2018_train`, or
  `prepare-data --ham10000-metadata HAM10000_metadata.csv` for the Harvard Dataverse / Kaggle release.
- **The default model was trained on these images**, so metrics on them measure training-set fit. The
  tool prints a warning.

### Your own dataset

Provide a CSV with `image_path,label` and optional `image_id,lesion_id,patient_id`. Labels must use the
canonical codes above. Point `dataset.metadata_csv` and `dataset.image_root` at it, then run `validate-data`.

### Licences and attribution

- ISIC 2018 / HAM10000 images: **CC-BY-NC**. Attribution: Tschandl, P., Rosendahl, C. & Kittler, H.
  *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin
  lesions.* Sci. Data 5, 180161 (2018); Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection
  2018: A Challenge Hosted by the ISIC.* arXiv:1902.03368. `ATTRIBUTION.txt`/`LICENSE.txt` are kept
  next to the downloaded images.
- Images and weights are **never committed** to this repository (`.gitignore`).
