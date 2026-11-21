# Command Cheatsheet

Every command, what it does, when to use it, and what its output looks like. All example outputs below
are **real output** captured from this project on the ISIC 2018 Validation set (trimmed where noted).

Run everything from the project folder:

```bash
cd ~/Workspace/Projects/AI-Skin
```

Two equivalent ways to call a command:

```bash
uv run skin-research <command>            # recommended: always uses the project's Python 3.11
python -m skin_research.cli <command>     # same thing, after `source .venv/bin/activate`
```

---

## 1. "I want to..." quick lookup

| I want to... | Command |
|---|---|
| check my setup works | `uv run skin-research doctor` |
| see a quick live result to show someone | `uv run skin-research demo --image-id ISIC_0034329` |
| test the model on my own dermoscopy image | `uv run skin-research demo --image path/to/img.jpg --label mel` |
| get predictions for a few images | `uv run skin-research predict img1.jpg img2.jpg` |
| check image quality of some photos | `uv run skin-research assess-quality path/to/folder` |
| run a short trial experiment (~1-5 min) | `uv run skin-research -e trial experiment --max-images 20 --conditions gaussian_blur enhance__clahe` |
| browse the results in a website | `uv run skin-research build-site`, then `explorer.exe "$(wslpath -w outputs/site/index.html)"` |
| run the full experiment (~30 min) | `uv run skin-research experiment` |
| rebuild charts/report after editing report code | `uv run skin-research report` |
| reproduce everything from scratch | `bash scripts/reproduce.sh` (or `--quick`) |
| make sure nothing is broken | `uv run pytest` |
| see all commands and options | `uv run skin-research --help` / `uv run skin-research <command> --help` |

---

## 2. First-time setup (run once, in this order)

| Step | Command | What it does | Time / size |
|---|---|---|---|
| 1 | `uv sync` | Creates `.venv` with Python 3.11 and the exact package versions from `uv.lock` | ~2 min, 1.4 GB |
| 2 | `uv run skin-research doctor` | Checks Python, packages, RAM, disk, data and model | seconds |
| 3 | `uv run skin-research download-data` | Downloads ISIC 2018 Validation images + labels into `data/raw/isic2018_val/` | 51 MB |
| 4 | `uv run skin-research prepare-data --fetch-lesion-ids` | Converts labels to `data/processed/isic2018_val_metadata.csv` and looks up lesion IDs | ~30 s |
| 5 | `uv run skin-research validate-data` | Checks every image and label | ~10 s |
| 6 | `uv run skin-research download-model` | Downloads and verifies the 16 MB model into `models/` | ~10 s |

After this everything works offline.

---

## 3. Every command in detail

### `doctor`: environment check
**Use it:** first thing on a new machine, or when something stops working.

```
skin-research 0.1.0 — environment check

  [OK] Python                 3.11.16 (.../AI-Skin/.venv/bin/python)
  [OK] Platform               Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
  [OK] CPU cores              6
  [OK] RAM available          7.8 GiB of 10.7 GiB
  [OK] Disk free              853 GiB
  [OK] torch                  2.15.0+cpu (threads=3)
  [OK] torchvision            0.29.2+cpu
  [OK] opencv                 4.14.0
  [OK] CUDA                   not available (CPU mode, expected)
  [OK] Config                 .../AI-Skin/configs/config.yaml
  [OK] Dataset metadata       data/processed/isic2018_val_metadata.csv
  [OK] Model weights          adeIaide/skin-lesion-7class-efficientnet cached

All checks passed.
```
`[!!]` marks something to fix. "CUDA not available" is expected: no GPU is needed.

### `download-data`: get a dataset
**Use it:** once per dataset. Options: `--list`, `--dataset KEY`, `-y` (skip the confirmation question).

```
$ uv run skin-research download-data --list
  isic2018_val       50.8 MB  ISIC 2018 Task 3 Validation: 193 dermoscopic images, held out from training
  isic2018_test     401.1 MB  ISIC 2018 Task 3 Test: 1,512 dermoscopic images, held out from training
  isic2018_train      2.6 GB  ISIC 2018 Task 3 Training (= HAM10000): 10,015 images with lesion groupings
```
Downloads over 100 MB ask `Continue? [y/N]` first. Don't evaluate on `isic2018_train`: the model was trained
on those images, so results would look unrealistically good.

### `prepare-data`: build the label CSV
**Use it:** right after `download-data`. `--fetch-lesion-ids` adds lesion IDs from the ISIC API so
photos of the same lesion are grouped together in the statistics.

```
Wrote 193 rows → .../data/processed/isic2018_val_metadata.csv
label
nv       123
bkl       22
mel       21
bcc       15
akiec      8
vasc       3
df         1
lesion_id available for 193/193 rows
```
For HAM10000 downloaded by hand: `prepare-data --ham10000-metadata path/HAM10000_metadata.csv --output data/processed/ham10000_metadata.csv`.

### `validate-data`: check the dataset
**Use it:** after preparing data, and whenever you add your own images. Exit code 1 means problems were found.

```
{
  "rows": 193,
  "valid_images": 193,
  "missing_files": 0,
  "unreadable_images": 0,
  "invalid_labels": 0,
  "exact_duplicate_groups": 0,
  "near_duplicate_pairs": 2,
  "issues": [
    "INFO: 2 near-duplicate image pair(s) (dHash distance ≤ 4); 1 with different labels. Same-lesion photos are expected in ISIC/HAM10000."
  ]
}
Per-image validation table → .../data/processed/isic2018_val_validation.csv
```
`near_duplicate_pairs` are for a human to look at, not errors. One of the two pairs here was a false alarm.

### `dataset-info`: class balance and grouping
**Use it:** to understand the data before interpreting results.

```
  akiec       8    4.2%  Actinic keratosis / intraepithelial carcinoma (Bowen's disease)
  bcc        15    7.8%  Basal cell carcinoma
  bkl        22   11.4%  Benign keratosis-like lesion
  df          1    0.5%  Dermatofibroma
  mel        21   10.9%  Melanoma
  nv        123   63.7%  Melanocytic nevus
  vasc        3    1.6%  Vascular lesion

{ "n_images": 193, "imbalance_ratio": 123.0, "n_lesions": 145, "split_level": "lesion", ... }
```

### `assess-quality`: image quality metrics
**Use it:** to measure brightness, contrast, blur and noise. With no arguments it checks the whole
dataset; give files or folders to check your own images.

```
$ uv run skin-research assess-quality data/raw/isic2018_val/ISIC2018_Task3_Validation_Input/ISIC_0034329.jpg
                     count       mean ...
brightness             1.0     0.7397      (0 = black, 1 = white)
rms_contrast           1.0     0.1825      (higher = more contrast)
laplacian_var          1.0  1136.7927      (blur score: higher = sharper)
tenengrad              1.0  1514.2579      (second sharpness score)
noise_sigma            1.0     1.1441      (estimated noise level)
dark_clipped           1.0     0.0000      (share of pure-black pixels)
bright_clipped         1.0     0.0001      (share of blown-out white pixels)
Flags: none
Report → .../outputs/reports/quality_custom.md
```
*(annotations in brackets added for this cheatsheet)*. The blur score is only useful for comparing an image
with its own blurred versions. Across real photos it varies 50×, and noise makes it go *up*.

### `generate-variations`: make degraded / enhanced images
**Use it:** to look at the transformed images without running the model. They are saved to
`data/variations/<experiment>/<condition>/`.

```
$ uv run skin-research generate-variations --max-images 7 --conditions gaussian_blur__moderate enhance__clahe
Generating 3 condition(s) for 15 image(s)
                         images  brightness  contrast  laplacian_var  questionable
original                     15       0.651     0.077        497.439             0
gaussian_blur__moderate      15       0.651     0.073         11.693            14
enhance__clahe               15       0.632     0.113       2371.148             0

Manifest (45 rows) → .../data/variations/<experiment>/manifest.csv
```
`--max-images 7` gave 15 images because it keeps whole lesions and at least one lesion per class.
`questionable` = images whose quality flags suggest the diagnosis may no longer be visible.

### `download-model`: get and verify the AI model
**Use it:** once. It prints the model's details and checks that its labels match the dataset.

```
  "repo_id": "adeIaide/skin-lesion-7class-efficientnet",
  "revision": "b5ebe73c3b89baa90ee4f991afa318385414a973",
  "architecture": "EfficientNetB0",
  "input_size": 320,
  "parameters": 4016515,
  "labels": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
  "weights_sha256": "fed0c3cb0a15823f4f6a82d4dd2725a0b8e6068a971664e1504403fba5b54562",
  "license": "CC-BY-NC-4.0"
Label mapping to dataset: OK
```

### `predict`: run the model
**Use it:** with image paths for quick predictions, or with no paths to predict every image listed in
the manifest from `generate-variations`.

```
$ uv run skin-research predict ISIC_0034329.jpg ISIC_0034323.jpg
 path              predicted_label  confidence  margin  prob_akiec  prob_bcc  prob_bkl  prob_df  prob_mel  prob_nv  prob_vasc
 .../ISIC_0034329.jpg         mel      0.8949  0.8579      0.0096    0.0096    0.0021   0.0371    0.8949   0.0131     0.0336
 .../ISIC_0034323.jpg         bcc      0.8933  0.8481      0.0452    0.8933    0.0116   0.0214    0.0014   0.0032     0.0239
2 predictions (2.6 img/s) → .../outputs/predictions/adhoc_predictions.csv

$ uv run skin-research predict          # no paths: uses the manifest
Predicting 45 images from .../manifest.csv
original                   0.933        (accuracy per condition)
gaussian_blur__moderate    0.533
enhance__clahe             0.733
```
- `confidence` = probability of the top class.
- `margin` = gap to the second most likely class; a small margin means the model is unsure.
- The 7 `prob_*` columns add up to 1.

### `experiment`: the full pipeline
**Use it:** to answer the research question. It runs data validation, image generation, predictions,
statistics, charts and the report in one go.

```
$ uv run skin-research -e cheatsheet_demo experiment --max-images 7 --conditions gaussian_blur__moderate enhance__clahe
Experiment 'cheatsheet_demo': 15 images × 3 conditions = 45 predictions in 15 s (8.57 img/s inference)
  report: .../outputs/reports/cheatsheet_demo/report.md

$ uv run skin-research experiment       # full run
Experiment 'isic2018_val_quality': 193 images × 41 conditions = 7913 predictions in 1715 s (8.55 img/s inference)
```
| Option | Use |
|---|---|
| `--max-images N` | quick runs on a smaller, class-balanced subset |
| `--conditions A B` | only some conditions (exact name or prefix, see section 5) |
| `--batch-size B` | lower it (e.g. 4) if memory is tight |
| `--force` | needed after you change a degradation/enhancement setting in the config |
| `--no-evaluate` | only generate images and predictions, skip statistics |

Always add `-e <name>` before `experiment` for trial runs. Without it a trial replaces the full study's
results (they can be restored by running the full `experiment` again; generated images are reused).

### `evaluate`: statistics from saved predictions
**Use it:** to recompute metrics without re-running the model (seconds instead of 30 minutes).

```
              condition  accuracy  balanced_accuracy  balanced_accuracy_ci_low  balanced_accuracy_ci_high  macro_f1  macro_roc_auc   ece
               original     0.933              0.952                     0.916                      1.000     0.943          0.995 0.201
gaussian_blur__moderate     0.533              0.500                     0.300                      0.833     0.441          0.699 0.477
         enhance__clahe     0.733              0.786                     0.625                      1.000     0.700          0.950 0.202
```
*(15-image trial run; full results are in the README.)*

### `report`: rebuild the Markdown report
**Use it:** after changing chart or report code, or to regenerate a deleted report.

```
Report → .../outputs/reports/<experiment>/report.md
Charts → .../outputs/charts/<experiment>
```

### `build-site`: interactive explainer website
**Use it:** after an experiment, to browse and show the results. It builds one self-contained page plus
small image thumbnails in `outputs/site/`. Every number comes from the experiment's output files.

```
$ uv run skin-research build-site
Website → .../outputs/site/index.html
Open it in a browser, e.g. from WSL: explorer.exe "$(wslpath -w .../outputs/site/index.html)"
```
Sections: the answer, how the pipeline works, a lesion explorer (original vs any condition with the
model's probabilities), all results (filter and sort), severity charts, enhancement charts, sensitivity per
diagnosis, findings, commands with copy buttons, tests and a glossary. To share it, zip the
`outputs/site` folder; it works offline.

### `demo`: one image, five versions
**Use it:** to show someone what the project does in 10 seconds.

```
$ uv run skin-research demo --image-id ISIC_0034329
Image ISIC_0034329  ground truth: mel  model: adeIaide/skin-lesion-7class-efficientnet

               version  brightness  rms_contrast  laplacian_var predicted_label  confidence
              original       0.740         0.183       1136.793             mel       0.895
gaussian_blur_moderate       0.740         0.179         13.665              nv       0.376
poor_lighting_moderate       0.393         0.126        567.198             mel       0.760
             sharpened       0.740         0.186       3832.022             mel       0.855
                 clahe       0.694         0.175       3885.031             mel       0.847

Comparison with original:
  gaussian_blur_moderate   CHANGED → nv       p(true=mel) 0.895 → 0.013
  poor_lighting_moderate   SAME class         p(true=mel) 0.895 → 0.760
  sharpened                SAME class         p(true=mel) 0.895 → 0.855
  clahe                    SAME class         p(true=mel) 0.895 → 0.847

Saved → .../outputs/demo/ISIC_0034329/demo_predictions.csv
```
The images are saved in `outputs/demo/<image_id>/`. Other interesting images: `--image-id ISIC_0034323` (bcc).
Your own image: `--image path.jpg --label mel` (the label is optional).

---

## 4. Global options (work with every command)

| Option | Use |
|---|---|
| `-e NAME` / `--experiment-name NAME` | write results under a different experiment name, so trial runs never overwrite the full study |
| `-c path/to/config.yaml` | use a different config file |
| `--log-level DEBUG` | show detailed logs when debugging |
| `--version` | print version |
| `-h` / `--help` | help for the CLI or for one command |

Full logs of every run are appended to `outputs/logs/skin_research.log`. Watch a running experiment with
`tail -f outputs/logs/skin_research.log` (Ctrl+C stops watching, not the experiment).

---

## 5. Condition names (for `--conditions`)

A name selects exactly that condition; a prefix selects a whole family. `original` is always included.

| Family (prefix) | Conditions |
|---|---|
| `gaussian_blur` | `gaussian_blur__mild`, `__moderate`, `__severe` |
| `motion_blur` | `motion_blur__mild`, `__moderate`, `__severe` |
| `underexposure` (poor lighting) | `underexposure__mild`, `__moderate`, `__severe` |
| `overexposure` | `overexposure__mild`, `__moderate`, `__severe` |
| `low_contrast` | `low_contrast__mild`, `__moderate`, `__severe` |
| `low_resolution` | `low_resolution__mild`, `__moderate`, `__severe` |
| `jpeg_compression` | `jpeg_compression__mild`, `__moderate`, `__severe` |
| `gaussian_noise` | `gaussian_noise__mild`, `__moderate`, `__severe` |
| `enhance__` | `enhance__brightness_correction`, `enhance__contrast_stretch`, `enhance__clahe`, `enhance__sharpen`, `enhance__denoise` |
| degrade → enhance | `gaussian_blur__moderate+sharpen`, `gaussian_blur__moderate+deblur`, `motion_blur__moderate+sharpen`, `underexposure__moderate+brightness_correction`, `underexposure__moderate+clahe`, `overexposure__moderate+brightness_correction`, `low_contrast__moderate+contrast_stretch`, `low_contrast__moderate+clahe`, `low_resolution__moderate+sharpen`, `gaussian_noise__moderate+denoise`, `jpeg_compression__moderate+denoise` |

Note: the prefix `gaussian_blur` also selects the two `gaussian_blur__moderate+...` combinations. The
strength of each level is set in `configs/config.yaml` under `degradations:` and `enhancements:`.

---

## 6. Testing

### Test commands

| Command | What it runs | Real result | Use it when |
|---|---|---|---|
| `uv run pytest` | all 72 tests, incl. real model + real data | `72 passed in 47.90s` | before committing / sharing |
| `uv run pytest -q` | same, compact output | `72 passed` | everyday check |
| `uv run pytest -m "not model and not dataset"` | only fast unit tests (no downloads needed) | `70 passed, 2 deselected in 13.31s` | quick check while editing code |
| `uv run pytest tests/test_degradation.py` | one test file | `14 passed in 0.42s` | after changing one module |
| `uv run pytest -k blur` | tests whose name contains "blur" | `6 passed, 66 deselected` | checking one feature |
| `uv run pytest -rs` | also print *why* tests were skipped | | if you see `skipped` |
| `uv run pytest -x` | stop at the first failure | | debugging |
| `uv run pytest -v` | one line per test with its name | | seeing exactly what ran |
| `uv run pytest --collect-only -q` | list tests without running them | `72 tests collected` | browsing the tests |

### What each test file checks

| File | Tests | Checks |
|---|---|---|
| `test_config.py` | 4 | config loads, bad settings rejected, paths resolve, config snapshot saves |
| `test_data.py` | 9 | loading, missing/corrupt files, bad labels, duplicates, lesion-level splits, never overwriting, zip safety |
| `test_quality.py` | 7 | blur/brightness/contrast/noise scores move the right way, edge cases, corrupt files |
| `test_degradation.py` | 14 | every degradation at every severity, blur gets worse with severity, noise is reproducible, input never modified |
| `test_enhancement.py` | 12 | every enhancement works and improves the metric it targets, colour preserved |
| `test_variations.py` | 5 | manifest traces every image to its source, reruns are identical, changed settings need `--force` |
| `test_inference.py` | 6 | label mapping, probability ordering, clear error without weights, **real model loads and predicts** |
| `test_evaluation.py` | 12 | metrics match scikit-learn, sensitivity/specificity, calibration, McNemar, Holm, lesion bootstrap |
| `test_report.py` | 2 | report has all sections, chart links work, no improvement claimed without evidence |
| `test_end_to_end.py` | 1 | **full pipeline on real ISIC images with the real model** (~1 min) |

### Reading test output

```
........................................................................ [100%]
72 passed in 47.90s
```
- `.` = passed, `F` = failed, `s` = skipped, `E` = error during setup.
- **skipped** means the test needs data or weights that aren't downloaded. Run setup steps 3–6, then retry.
  A skipped test is *not* a pass.
- On failure, pytest prints the failing line and the values involved. Run that one file again with
  `uv run pytest tests/<file>.py -x -v`.

---

## 7. Where outputs go

| Path | Contents | Made by |
|---|---|---|
| `outputs/reports/<exp>/report.md` | the research report (open in VS Code, Ctrl+Shift+V) | experiment, evaluate, report |
| `outputs/reports/<exp>/metrics_summary.csv` | one row per condition with all metrics and CIs | experiment, evaluate |
| `outputs/reports/<exp>/metrics_vs_original.csv` | change vs original, flip rate, McNemar p | experiment, evaluate |
| `outputs/reports/<exp>/metrics_per_class.csv` | sensitivity/specificity per class per condition | experiment, evaluate |
| `outputs/reports/<exp>/metrics_restoration.csv` | degraded vs degraded→enhanced | experiment, evaluate |
| `outputs/reports/<exp>/metrics.json` | everything above in one file | experiment, evaluate |
| `outputs/charts/<exp>/*.png` | 9 charts | experiment, evaluate, report |
| `outputs/predictions/<exp>/predictions.csv` | every prediction with probabilities and quality scores | experiment, predict |
| `outputs/predictions/<exp>/run_info.json` | model hash, versions, timings | experiment |
| `outputs/predictions/<exp>/config_used.yaml` | exact settings of that run | experiment |
| `data/variations/<exp>/<condition>/*.png` | the generated images | experiment, generate-variations |
| `data/variations/<exp>/manifest.csv` | which original + settings produced each image | experiment, generate-variations |
| `outputs/demo/<image_id>/` | demo images + CSV | demo |
| `outputs/reports/quality_*.md/.csv` | quality reports | assess-quality |
| `outputs/logs/skin_research.log` | full log of all runs | every command |

`<exp>` is `experiment.name` in the config (default `isic2018_val_quality`). Open a folder in Windows
Explorer with `explorer.exe outputs/charts/isic2018_val_quality`.

---

## 8. Common errors and fixes

| You see | Meaning | Fix |
|---|---|---|
| `Metadata CSV not found: ...` | the label CSV doesn't exist yet | `download-data`, then `prepare-data` |
| `Unknown condition(s) ['fog']. Known: original, gaussian_blur__mild, ...` | typo in `--conditions` | use a name from section 5 |
| `... was generated with different parameters. Re-run with --force ...` | you changed a severity setting | add `--force`, or change `experiment.name` to keep both |
| `Could not obtain 'model.safetensors' ...` | model download failed | check internet, rerun `download-model` |
| `Refusing to compute metrics — label mapping incompatible` | model and dataset classes don't match | fix `model.label_map` in the config |
| `No predictions at ... Run skin-research experiment first.` | `evaluate`/`report` before any experiment | run `experiment` |
| `Configuration error: ...` | a typo in the YAML config | the message names the bad section |
| experiment seems frozen | PC went to sleep, or it's on a slow step | check `tail outputs/logs/skin_research.log`; keep Windows awake |
| `python: command not found` | not using the project Python | put `uv run` in front |

A run that is interrupted (Ctrl+C or a crash) can be restarted with the same command: images that were
already generated are reused.
