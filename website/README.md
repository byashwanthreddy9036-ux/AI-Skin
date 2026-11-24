# Presentation website

An interactive, evidence-based presentation of the skin lesion image-quality study, for audiences with no
background in machine learning or medical imaging. Plain HTML, CSS and JavaScript: no framework, no build
step, no server required.

## Start it

From the project folder:

```bash
uv run skin-research build-site     # export the latest experiment results into website/data and website/assets
uv run skin-research serve-site     # then open http://127.0.0.1:8000   (Ctrl+C to stop)
```

Or skip the server and open `website/index.html` directly. From WSL:

```bash
explorer.exe "$(wslpath -w website/index.html)"
```

Re-run `build-site` whenever an experiment finishes (for example the ISIC Test set); the site picks up the
new results and status automatically. Fonts load from Google Fonts when online and fall back to system fonts
offline.

| Link | Opens |
|---|---|
| `index.html#overview` … `#about` | a section directly (`overview`, `how`, `lab`, `explorer`, `results`, `about`) |
| `index.html#tour` / `#tour4` | the guided tour, at step 1 / step 4 |

## Sections

1. **Overview**: the research question, the experiment shown on one real photo, key facts and findings.
2. **How It Works**: eight clickable pipeline stages (what, how, why, output), the seven diagnoses, and how
   EfficientNet decides.
3. **Image Lab**: the main experience. A curated gallery of real cases, then any lesion × any transformation
   × mild/moderate/severe, shown side by side or with a before/after slider, with both predictions, confidence,
   true diagnosis, whether the prediction changed, quality measurements and all seven probabilities.
4. **Prediction Explorer**: one lesion across all 41 conditions, plus why confidence is not accuracy.
5. **Results**: nine question-led charts, a switch between experiments, and status cards for experiments
   that are not finished.
6. **About**: dataset, preparation, model, transformations, design, statistics, reproducibility,
   limitations, ethics and references.

**Guided tour** (button in the top bar): seven full-screen steps for presenting. Use **← / →** to move and
**Esc** to exit. See [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md).

## How it is built

```
outputs/  (experiment results, read-only)  ──►  skin-research build-site  ──►  website/data/site-data.js
data/variations/ (generated images)                (src/skin_research/          website/data/SAMPLES.md
configs/website.yaml (which experiments)            website/export.py)          website/assets/samples/<id>/*.jpg
                                                                                        │
                                                    website/index.html + css + js ◄─────┘  (read SITE_DATA only)
```

| Layer | Location | Notes |
|---|---|---|
| Research computation | `src/skin_research/` (pipeline) | not changed by the website |
| Experiment outputs | `outputs/`, `data/variations/` | only read |
| Presentation data | `website/data/`, `website/assets/samples/` | generated, git-ignored |
| Frontend | `website/index.html`, `css/`, `js/` | committed; one file per section |

Settings (`configs/website.yaml`): which experiments to show and in which order, which experiment feeds the
Image Lab (`primary`), how many lesions to export, and image sizes. Paths are relative to the project root.

### Experiment status

An experiment's numbers are exported only if it is verifiably **complete**: its `status.json` is not
`running`/`failed`, `predictions.csv`, `run_info.json` and all metric tables exist, predictions = images ×
conditions, every condition was evaluated, every dataset image was evaluated (a trial subset counts as
*incomplete*), and the outputs belong to the configured dataset. Otherwise the site shows the status
(*dataset not prepared*, *not started*, *running* with progress, *interrupted*, *failed* or *incomplete*)
and no numbers. The validation-vs-test comparison appears only when both are complete; the two datasets are
never pooled.

## Where every number comes from

All files are inside `outputs/` for the experiment named in the chart footnote (validation:
`isic2018_val_quality`, test: `isic2018_test_quality`). Every chart on the site prints its source.

| Shown on the site | File | Columns / keys |
|---|---|---|
| Balanced accuracy, accuracy and their 95% CIs; macro F1, ROC-AUC, specificity | `reports/<exp>/metrics_summary.csv` | `balanced_accuracy`, `*_ci_low/high`, `accuracy`, `macro_f1`, `macro_roc_auc`, `macro_specificity` |
| Change vs original, "measurably lower/higher", flip rate, McNemar | `reports/<exp>/metrics_vs_original.csv` | `delta_balanced_accuracy`, `delta_balanced_accuracy_ci_*`, `flip_rate`, `mcnemar_p_holm` |
| Sensitivity per diagnosis, melanoma detection, heatmap | `reports/<exp>/metrics_per_class.csv` | `recall` (= sensitivity), `specificity`, `support` |
| Damaged → fixed, recovery | `reports/<exp>/metrics_restoration.csv` | `balacc_degraded`, `balacc_enhanced`, `recovery_fraction`, delta CIs |
| Confusion matrices, reliability diagrams | `reports/<exp>/metrics.json` | `per_condition.<condition>.confusion_matrix`, `.reliability_bins` |
| Confidence (calibration) | `reports/<exp>/metrics_summary.csv` | `mean_confidence`, `mean_conf_correct`, `mean_conf_wrong`, `ece`, `brier` |
| Confidence histograms (right vs wrong) | `predictions/<exp>/predictions.csv` | `confidence`, `correct`, counted per 10% bin |
| Every individual prediction and probability (Lab, Explorer, gallery, tour) | `predictions/<exp>/predictions.csv` | `predicted_label`, `confidence`, `margin`, `prob_*`, `correct` |
| Image quality per photo | `predictions/<exp>/predictions.csv` | `brightness`, `rms_contrast`, `laplacian_var`, `noise_sigma`, `quality_flags` |
| Images | `output_path` of each row in `predictions.csv` (original photos in `data/raw/…`, transformed ones in `data/variations/<exp>/…`) | resized to 520 px JPEG for display |
| Dataset, model and run facts | `predictions/<exp>/run_info.json` | `n_images`, `n_conditions`, model metadata, environment |
| Findings text in reports | `reports/<exp>/report.md` | section 7 |
| Transformation settings | `configs/config.yaml` | `degradations`, `enhancements` |

The 520 px display images are JPEG copies made for the browser; the model was evaluated on the full-size,
losslessly saved originals.

## Real image IDs used

All from the ISIC 2018 Task 3 **validation** set, chosen by the rules in `export.py` (`STORY_RULES`).
`build-site` rewrites the authoritative list to `website/data/SAMPLES.md` each time. The list at the time of
writing:

| Curated case | Image ID | True diagnosis | Condition shown |
|---|---|---|---|
| Correct on the original photo | ISIC_0034329 | melanoma | original |
| Blur changes the diagnosis | ISIC_0034446 | melanoma | Gaussian blur, mild |
| Poor lighting changes the diagnosis | ISIC_0034477 | melanoma | underexposure, moderate |
| Brightness correction rescues it | ISIC_0034402 | actinic keratosis (akiec) | underexposure (moderate) → brightness correction |
| Enhancement makes it worse | ISIC_0034394 | melanoma | CLAHE |
| Confident, but wrong | ISIC_0034398 | melanoma | low contrast, moderate |
| Sensor noise or compression changes the diagnosis | ISIC_0034492 | melanoma | sensor noise, moderate |
| Robust to most damage | ISIC_0034376 | melanoma | original |

No melanoma qualified for the "brightness correction rescues it" rule, so an actinic keratosis case is shown.
Each lesion is exported with all 41 conditions (8 lesions, 336 images including thumbnails).

## Checking the site

```bash
uv run python scripts/check_website.py      # headless browser test at 1440, 768 and 375 px
uv run pytest tests/test_website.py         # exported data vs the experiment CSVs, status logic, case rules
```

`check_website.py` opens `index.html?selftest`, which clicks through every section, every Image Lab
combination (8 lesions × all transformations and strengths), every Explorer card, every Results control on
each experiment and every tour step. It checks that the predictions shown match the data, loads every image
and measures sideways overflow. `frame.html` shows the site at an exact phone width, because headless
browsers cannot open windows narrower than about 500 px.

## Files

```
website/
├── index.html           page shell, six sections, tour overlay
├── frame.html           developer tool: fixed-width preview (?w=375) for phone checks
├── css/site.css         design tokens and layout
├── js/util.js           data access, formatting, tooltips, SVG helpers
├── js/content.js        plain-language explanations (numbers filled in from data)
├── js/overview.js       sections 1 and 2
├── js/lab.js            section 3, Image Lab and gallery
├── js/explorer.js       section 4, Prediction Explorer
├── js/results.js        section 5, Results charts and status cards
├── js/about.js          section 6
├── js/tour.js           guided tour
├── js/app.js            navigation
├── js/selftest.js       in-browser test (only with ?selftest)
├── data/                generated: site-data.js, SAMPLES.md
└── assets/samples/      generated: display images and thumbnails
```
