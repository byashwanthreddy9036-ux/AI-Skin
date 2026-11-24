/* Plain-language explanations. Experiment numbers are filled in from SITE_DATA, never typed here. */
"use strict";

const C = {};

C.DIAGNOSES = {
  mel: { kind: "malignant", tag: "Skin cancer", text: "Melanoma. The most dangerous common skin cancer; early detection strongly affects survival." },
  bcc: { kind: "malignant", tag: "Skin cancer", text: "Basal cell carcinoma. The most common skin cancer; grows slowly and rarely spreads." },
  akiec: { kind: "precancer", tag: "Pre-cancer / early cancer", text: "Actinic keratosis or Bowen's disease. Sun-damage lesions that can progress to squamous cell carcinoma." },
  nv: { kind: "benign", tag: "Benign", text: "Melanocytic nevus: an ordinary mole. By far the most common lesion in the data." },
  bkl: { kind: "benign", tag: "Benign", text: "Benign keratosis: seborrhoeic keratoses, solar lentigines and similar harmless growths." },
  df: { kind: "benign", tag: "Benign", text: "Dermatofibroma. A small, firm, harmless bump in the skin." },
  vasc: { kind: "benign", tag: "Benign", text: "Vascular lesion: cherry angiomas and other blood-vessel growths." },
};

C.TRANSFORMS = {
  gaussian_blur: "An out-of-focus photo. Each pixel is averaged with its neighbours; σ is the blur radius in pixels.",
  motion_blur: "Camera shake. Pixels are smeared along a line of the given length and angle.",
  underexposure: "Too little light. Mid-tones are darkened with a gamma curve and overall brightness is reduced.",
  overexposure: "Too much light or flash. Brightness is multiplied, and bright areas clip to pure white and lose detail.",
  low_contrast: "A washed-out photo. All tones are pulled towards the average grey.",
  low_resolution: "A low-resolution camera. The photo is shrunk and enlarged back; the lost detail does not return.",
  jpeg_compression: "Heavy file compression. Creates 8×8-pixel blocks and halos around edges.",
  gaussian_noise: "Sensor grain from a small camera or dim light. Random noise is added to every pixel.",
  brightness_correction: "Moves the average lightness to a target level with a gamma curve; colours are kept.",
  contrast_stretch: "Stretches the darkest 1% of tones to black and the brightest 1% to white.",
  clahe: "Contrast-Limited Adaptive Histogram Equalisation: boosts local contrast in small tiles.",
  sharpen: "Unsharp mask: adds back the difference between the photo and a blurred copy, making edges crisper.",
  denoise: "Non-local means: averages similar-looking patches to remove grain. It can also erase fine texture.",
  deblur: "Richardson–Lucy deconvolution: tries to reverse a blur of an assumed size. Only works if the guess is right.",
};

C.STAGES = () => {
  const v = U.primary(), m = v.meta;
  const nDeg = new Set(D.catalog.filter(c => c.group === "degradation").map(c => c.degradation)).size;
  const nEnh = D.catalog.filter(c => c.group === "enhancement").length;
  const nFix = D.catalog.filter(c => c.group === "degrade_enhance").length;
  const val = m.validation || {};
  return [
    { k: "Dataset collection", what: `${U.int(m.n_images)} expert-labelled dermoscopic photos from the ISIC 2018 challenge (validation set), covering ${U.int(m.n_lesions)} different lesions and 7 diagnoses.`,
      how: "Downloaded from the official ISIC archive. Lesion identifiers were looked up from the public ISIC API so photos of the same lesion can be kept together.",
      why: "The model was trained on the separate ISIC training set (HAM10000). Testing on photos it never saw is the only fair test; testing on training photos would be like grading students on memorised questions.",
      out: "A labels table: one row per photo with its file, diagnosis and lesion ID.", cmd: "skin-research download-data · prepare-data" },
    { k: "Image validation", what: `Every photo is opened and checked: ${U.int(val.valid_images)} of ${U.int(val.rows)} valid, ${U.int(val.missing_files)} missing, ${U.int(val.unreadable_images)} corrupt, ${U.int(val.near_duplicate_pairs)} near-duplicate pairs flagged for review.`,
      how: "Files are decoded completely, labels are checked against the 7 allowed codes, exact copies are found by file fingerprint (SHA-256) and near-copies by a perceptual hash.",
      why: "A missing file or a wrong label silently corrupts every result that follows. Problems are reported, never ignored.",
      out: "A validation table and a summary of any problems.", cmd: "skin-research validate-data" },
    { k: "Quality assessment", what: "Each photo (and later each transformed copy) gets objective quality scores: brightness, contrast, blur, noise and clipped highlights.",
      how: "Classical image measurements. Blur, for example, is the variance of the Laplacian: how strongly edges stand out after an edge-detection filter.",
      why: "To describe what each transformation did to the photo in numbers, and to flag photos whose diagnosis may no longer be visible. These scores are heuristics, not a verdict.",
      out: "Quality scores per image and per condition.", cmd: "skin-research assess-quality" },
    { k: "Image degradation", what: `${nDeg} kinds of realistic damage, each at mild, moderate and severe strength: ${nDeg * 3} damaged versions of every photo.`,
      how: "Simple, repeatable image operations (blur kernels, gamma curves, JPEG re-encoding, random noise with a fixed seed). The original files are never changed.",
      why: "Real clinic and phone photos are often blurry, dark or compressed. Controlled damage shows exactly how much each problem costs.",
      out: "Damaged copies of every photo plus a manifest linking each copy to its source and settings.", cmd: "skin-research generate-variations" },
    { k: "Image enhancement", what: `${nEnh} classical enhancements applied to good photos, and ${nFix} damage-then-fix pairs (for example, darkened then brightness-corrected).`,
      how: "Traditional filters only: brightness correction, contrast stretch, CLAHE, sharpening, denoising, deblurring. No AI image generators.",
      why: "Generative 'enhancers' can invent lesion detail that is not there, which is unacceptable for medical images. We test whether simple fixes help or hurt.",
      out: "Enhanced copies, recorded in the same manifest.", cmd: "skin-research generate-variations" },
    { k: "AI prediction", what: `The same frozen model classifies every version: ${U.int(m.n_predictions)} predictions in total, each a probability for all 7 diagnoses.`,
      how: `${m.model.architecture} (${U.int(m.model.parameters)} parameters) on a regular CPU at about ${m.throughput} images per second. The weights are pinned to one exact version and verified on load.`,
      why: "Keeping the model and its settings identical means any change in the answer is caused by the image, not the model.",
      out: "A predictions table: one row per photo × condition with all 7 probabilities.", cmd: "skin-research predict · experiment" },
    { k: "Statistical evaluation", what: "Accuracy, balanced accuracy, sensitivity per diagnosis, calibration, and how often the answer changes compared with the original photo.",
      how: `${U.int(m.bootstrap)} bootstrap resamples of whole lesions give 95% confidence intervals; each condition is compared with the original on the very same photos; McNemar tests with Holm correction check many comparisons at once.`,
      why: `With only ${U.int(m.n_images)} photos, differences can appear by chance. Confidence intervals show which differences are real.`,
      out: "Metric tables (CSV and JSON) per condition, per diagnosis and per comparison.", cmd: "skin-research evaluate" },
    { k: "Report generation", what: "A written report, charts and this website are produced automatically from the result tables.",
      how: "Observations are written by fixed rules: an effect is only called real when its confidence interval excludes zero.",
      why: "Automatic reporting prevents hand-picked or mistyped numbers and makes every result traceable to a file.",
      out: "A Markdown report, chart images and the website data.", cmd: "skin-research report · build-site" },
  ];
};

C.NN_STEPS = [
  ["1 · Resize", "The photo is resized to 320 × 320 pixels and its colours are normalised the same way as during training."],
  ["2 · Edges", "Early layers respond to simple patterns: edges, colour changes, dots."],
  ["3 · Textures", "Middle layers combine them into textures such as pigment networks, globules and streaks."],
  ["4 · Patterns", "Deep layers recognise lesion-level patterns that distinguish the 7 diagnoses."],
  ["5 · 7 scores", "A final layer turns them into 7 probabilities that add up to 100%. The largest is the prediction."],
];

C.ABOUT_REFS = [
  "Tschandl P., Rosendahl C., Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161 (2018).",
  "Codella N. et al. Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368 (2019).",
  "Tan M., Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML (2019).",
  "Pizer S. M. et al. Adaptive histogram equalization and its variations. Computer Vision, Graphics, and Image Processing 39 (1987) [CLAHE].",
  "Buades A., Coll B., Morel J.-M. A non-local algorithm for image denoising. CVPR (2005).",
  "Efron B., Tibshirani R. An Introduction to the Bootstrap. Chapman & Hall (1993).",
  "McNemar Q. Note on the sampling error of the difference between correlated proportions. Psychometrika 12 (1947); Holm S. A simple sequentially rejective multiple test procedure. Scand. J. Statistics 6 (1979).",
];

window.C = C;
