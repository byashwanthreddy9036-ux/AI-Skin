/* Section 6: About the research. Experiment-specific values are read from SITE_DATA. */
"use strict";

const About = {};

About.render = () => {
  const r = U.primary(), m = r.meta, md = m.model;
  const cat = D.catalog;
  const deg = cat.filter(c => c.group === "degradation");
  const types = [...new Set(deg.map(c => c.degradation))];
  const params = (d, s) => Object.entries((cat.find(c => c.name === `${d}__${s}`) || {}).degradation_params || {})
    .map(([k, v]) => `${k} ${v}`).join(", ");
  const enh = cat.filter(c => c.group === "enhancement");
  const counts = m.class_counts;
  const rep = md.reported_evaluation || {};
  const test = U.exp("test");
  const sections = [
    ["dataset", "ISIC 2018 dataset", `
      <p>The International Skin Imaging Collaboration (ISIC) 2018 challenge released dermoscopic photos with diagnoses
        confirmed by histopathology, follow-up or expert consensus. The model was trained on the challenge's training set
        (HAM10000). This study uses the separate <b>validation set (${U.int(m.n_images)} photos, ${U.int(m.n_lesions)} lesions)</b>,
        which the model never saw${test ? `, and the separate test set (1,512 photos), whose run is currently: <b>${U.esc(Results.statusText(test.status))}</b>` : ""}.</p>
      <p class="small muted">Photos per diagnosis (validation): ${D.labels.map(l => `${l} ${counts[l] ?? 0}`).join(" · ")}.</p>`],
    ["prep", "Dataset preparation", `<ul>
      <li>Downloaded from the official ISIC archive; original files are never modified (generated images go to a separate folder and existing files are never overwritten).</li>
      <li>Labels converted to one table; lesion IDs added from the public ISIC API so photos of the same lesion are grouped in statistics.</li>
      <li>Every file decoded and checked: ${U.int(m.validation.valid_images)} of ${U.int(m.validation.rows)} valid, ${U.int(m.validation.exact_duplicate_groups)} exact duplicates, ${U.int(m.validation.near_duplicate_pairs)} near-duplicate pairs flagged.</li>
      <li>The set of evaluated photos is fixed and saved before any image is changed (split level: ${U.esc(m.split_level)}).</li></ul>`],
    ["model", "Model architecture", `
      <p><b>${U.esc(md.architecture)}</b> with ${U.int(md.parameters)} parameters, input ${md.input_size} × ${md.input_size} pixels,
        7 output classes. Weights: <a href="${U.esc(md.source_url)}" target="_blank" rel="noopener">${U.esc(md.repo_id)}</a>, pinned to revision
        <code>${U.esc((md.revision || "").slice(0, 12))}</code>, SHA-256 <code>${U.esc((md.weights_sha256 || "").slice(0, 16))}…</code>, licence ${U.esc(md.license)}.</p>
      <p class="small muted">The model card reports, on its own lesion-disjoint development split: macro-F1 ${U.f3(rep.macro_f1)}, balanced accuracy
        ${U.f3(rep.balanced_accuracy)}, macro AUROC ${U.f3(rep.macro_auroc)}. These are the publisher's claims; this study measures the model itself.</p>`],
    ["transforms", "Image transformations", `
      <div class="scroll"><table class="t"><thead><tr><th>Damage</th><th>Simulates</th><th>Mild</th><th>Moderate</th><th>Severe</th></tr></thead><tbody>
      ${types.map(d => `<tr><td><b>${U.NAMES[d]}</b></td><td class="small">${C.TRANSFORMS[d]}</td>${["mild", "moderate", "severe"].map(s => `<td class="small mono">${U.esc(params(d, s))}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div>`],
    ["enhancement", "Enhancement methods", `
      <div class="scroll"><table class="t"><thead><tr><th>Method</th><th>What it does</th><th>Settings</th></tr></thead><tbody>
      ${enh.map(c => `<tr><td><b>${U.NAMES[c.enhancement]}</b></td><td class="small">${C.TRANSFORMS[c.enhancement]}</td><td class="small mono">${U.esc(U.params(c.name))}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small muted">Damage-then-fix pairs (moderate damage only): ${cat.filter(c => c.group === "degrade_enhance").map(c => U.esc(U.title(c.name))).join("; ")}.</p>`],
    ["design", "Experimental design", `<ul>
      <li><b>Paired design:</b> every condition uses exactly the same photos and the same frozen model with the same preprocessing, so differences come from the image change.</li>
      <li><b>${U.int(m.n_conditions)} conditions</b> × ${U.int(m.n_images)} photos = ${U.int(m.n_predictions)} predictions, on CPU (${m.throughput} photos per second).</li>
      <li>Labels are kept for damaged photos (the lesion did not change). Photos whose quality scores fall outside simple thresholds are flagged as "questionable" rather than dropped.</li>
      <li>All settings were fixed before results were seen; nothing was tuned on the evaluation photos.</li></ul>`],
    ["stats", "Statistical methods", `<ul>
      <li><b>Balanced accuracy</b> (primary), accuracy, per-diagnosis sensitivity and specificity, macro F1, ROC-AUC.</li>
      <li><b>${U.int(m.bootstrap)}-resample bootstrap of whole lesions</b> for ${Math.round(m.confidence_level * 100)}% confidence intervals; photos of one lesion are not independent.</li>
      <li><b>Paired comparisons:</b> the change versus the original uses the same resampled lesions for both; an effect is only reported when its interval excludes zero.</li>
      <li><b>McNemar's exact test</b> on per-photo correctness, with <b>Holm correction</b> for the many comparisons.</li>
      <li><b>Calibration:</b> expected calibration error (10 bins), Brier score, and the confidence of right versus wrong answers.</li></ul>`],
    ["repro", "Reproducibility", `<ul>
      <li>Fixed seed (${m.seed}); random noise per photo is derived from the photo ID, so reruns give identical images. A full rerun reproduced every metric exactly.</li>
      <li>Pinned package versions (<code>uv.lock</code>), pinned model revision and weights hash, Python ${U.esc(m.python)}.</li>
      <li>Each run saves its exact configuration, environment and code version (git <code>${U.esc((m.git_commit || "").slice(0, 8))}</code>), plus a manifest linking every generated image to its source and settings.</li>
      <li>Reproduce with <code>bash scripts/reproduce.sh</code>; rebuild this site with <code>uv run skin-research build-site</code>.</li></ul>`],
    ["limits", "Limitations", `<ul>
      <li>Small sample: ${U.int(m.n_images)} photos, with only ${counts.df ?? 0} dermatofibroma and ${counts.vasc ?? 0} vascular-lesion photos, so per-diagnosis numbers for rare classes are unstable.</li>
      <li>Damage is simulated. Real blur varies across a photo and real lighting also shifts colour.</li>
      <li>One model. Other models, or models trained with image augmentation, may be more or less robust.</li>
      <li>Dermoscopic photos only, from a dataset that under-represents darker skin tones; results do not transfer to phone photos or other populations.</li>
      <li>Patient IDs are not public, so photos of the same patient (but different lesions) could not be grouped.</li>
      <li>Quality warnings use simple thresholds that are not validated clinical criteria.</li></ul>`],
    ["ethics", "Ethical considerations", `
      <div class="notice"><b>This is a research experiment, not a diagnostic product.</b> It must not be used to assess anyone's skin. A
        concerning spot should be examined by a doctor.</div>
      <ul>
      <li>All photos are de-identified public research data used under their licence (CC-BY-NC) for non-commercial research and education.</li>
      <li>The findings argue for caution: an AI that silently degrades on poor photos, without lowering its confidence, can give false reassurance about a cancer.</li>
      <li>Under-representation of darker skin in the training and test data means performance for those patients is unknown.</li></ul>`],
    ["refs", "References and attribution", `
      <p class="small">Images: ISIC 2018 Task 3 (CC-BY-NC), International Skin Imaging Collaboration; HAM10000 by Tschandl et al.
        Model weights: ${U.esc(md.repo_id)} (${U.esc(md.license)}).</p>
      <ul class="small">${C.ABOUT_REFS.map(x => `<li>${U.esc(x)}</li>`).join("")}</ul>`],
  ];
  U.$("#s-about").innerHTML = `
    <div class="wrap">
      <div class="section-head"><span class="eyebrow">About the research</span><h2>Methods, limits and sources</h2>
        <p>The technical detail behind every number on this site.</p></div>
      <div class="about">
        <nav class="toc" aria-label="About sections">${sections.map(([id, t]) => `<a href="#about-${id}" data-local>${t}</a>`).join("")}</nav>
        <div class="about-body">${sections.map(([id, t, body]) => `<section id="about-${id}"><h3>${t}</h3>${body}</section>`).join("")}</div>
      </div>
    </div>`;
};

window.About = About;
