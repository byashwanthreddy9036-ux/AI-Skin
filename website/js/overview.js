/* Section 1 (Overview) and Section 2 (How It Works). */
"use strict";

const Overview = {};

/* Headline findings computed from the primary experiment's result tables. */
Overview.findings = () => {
  const key = D.primary, r = U.primary(), o = U.orig(key);
  const out = [];
  const others = r.conditions.filter(c => c.name !== "original");
  const worse = others.filter(c => c.d_hi != null && c.d_hi < 0);
  out.push({ big: `${worse.length} of ${others.length}`, text: "image changes made the AI measurably less accurate (95% confidence interval of the change entirely below zero).",
    src: "metrics_vs_original.csv · delta_balanced_accuracy_ci_high" });
  const mildBlur = U.cond(key, "gaussian_blur__mild");
  if (mildBlur && r.per_class.original && r.per_class.original.mel && r.per_class.gaussian_blur__mild) {
    out.push({ big: `${U.pct(r.per_class.original.mel.recall)} → ${U.pct(r.per_class.gaussian_blur__mild.mel.recall)}`,
      text: "of melanomas detected when the photo is only mildly out of focus.", src: "metrics_per_class.csv · recall (mel)" });
  }
  const enh = r.conditions.filter(c => U.catalog[c.name] && U.catalog[c.name].group === "enhancement");
  const enhBetter = enh.filter(c => c.d_lo != null && c.d_lo > 0);
  const enhWorse = enh.filter(c => c.d_hi != null && c.d_hi < 0);
  out.push({ big: `${enhBetter.length} of ${enh.length}`, text: `enhancements improved photos that were already good; ${enhWorse.length} made the AI measurably worse.`,
    src: "metrics_vs_original.csv" });
  const fix = r.restoration.find(x => x.name === "underexposure__moderate+brightness_correction");
  if (fix && fix.recovery != null) {
    out.push({ big: U.pct(fix.recovery), text: "of the accuracy lost to poor lighting was recovered by brightness correction, the clearest case of a fix that matches the damage.",
      src: "metrics_restoration.csv · recovery_fraction" });
  }
  return out;
};

Overview.render = () => {
  const key = D.primary, r = U.primary(), m = r.meta, o = U.orig(key);
  const story = D.samples.stories.find(s => s.key === "correct") || D.samples.stories[0];
  const les = U.lesion(story.image_id);
  const shown = ["original", "gaussian_blur__moderate", "underexposure__moderate", "enhance__clahe"]
    .filter(c => les.conditions[c] && les.conditions[c].img);
  const card = c => {
    const rec = les.conditions[c];
    return `<div class="flow-card"><img src="${rec.img}" alt="${U.esc(U.title(c))} of lesion ${les.id}" loading="lazy">
      <div class="cap"><b>${U.esc(U.shortTitle(c))}</b><span>${rec.correct ? "✓" : "✗"} ${rec.pred} ${U.pct(rec.conf)}</span></div></div>`;
  };
  const test = U.exp("test");
  const testText = !test ? "" : test.status.state === "complete"
    ? `${U.int(test.results.meta.n_images)} test photos`
    : `test set (${U.int(1512)} photos): ${Results.statusText(test.status)}`;
  const nTypes = new Set(D.catalog.filter(c => c.degradation).map(c => c.degradation)).size;
  const nEnh = new Set(D.catalog.filter(c => c.enhancement).map(c => c.enhancement)).size;

  U.$("#s-overview").innerHTML = `
    <div class="wrap">
      <div class="hero">
        <div class="stack">
          <span class="eyebrow">Research project · dermatology AI · image quality</span>
          <h1>Skin Lesion Image Quality &amp; AI Reliability</h1>
          <p class="lede">We investigate how changes in skin-lesion image quality affect the reliability of AI-based
            classification models, and whether traditional image enhancement techniques can recover lost performance.</p>
          <p class="question">Research question: when a skin-lesion photo is blurry, dark or compressed, does an AI classifier
            become less accurate, and can classic image fixes bring the accuracy back?</p>
          <div class="row" style="margin-top:6px">
            <a class="btn" href="#lab">Explore the Experiment →</a>
            <button class="btn ghost" type="button" data-tour>Start guided tour</button>
          </div>
        </div>
        <div class="flow" aria-label="The experiment in one picture">
          <span class="label">The experiment, on a real photo (${U.esc(les.id)}, true diagnosis: ${U.esc(U.label(les.label))})</span>
          <div class="flow-grid">${shown.map(card).join("")}</div>
          <p class="small muted">Same lesion, same AI model. Only the photo quality changes. Labels show the AI's answer and its confidence.</p>
        </div>
      </div>
      <div class="facts">
        <div class="fact"><span class="label">Dataset</span><b>${U.int(m.n_images)} photos</b><span class="small muted">ISIC 2018 validation set${testText ? `; ${U.esc(testText)}` : ""}</span></div>
        <div class="fact"><span class="label">AI model</span><b>${U.esc(m.model.architecture)}</b><span class="small muted">pretrained, 7 diagnoses, ${U.int(m.model.parameters)} parameters</span></div>
        <div class="fact"><span class="label">Transformations</span><b>${U.int(m.n_conditions)} conditions</b><span class="small muted">${nTypes} damage types × 3 strengths, ${D.catalog.filter(c => c.group === "enhancement").length} enhancements of good photos, ${D.catalog.filter(c => c.group === "degrade_enhance").length} damage-then-fix pairs (${nEnh} methods incl. deblurring)</span></div>
        <div class="fact"><span class="label">AI predictions</span><b>${U.int(m.n_predictions)}</b><span class="small muted">balanced accuracy on original photos ${U.f3(o.bal)}</span></div>
      </div>
      <div class="block">
        <h2>Key findings so far</h2>
        <p>From the completed ${U.int(m.n_images)}-photo validation experiment. Each number is read from the result files named under it.</p>
        <div class="findings-grid">${Overview.findings().map(f => `
          <div class="finding"><span class="big num">${f.big}</span><p>${f.text}</p><span class="src">Source: ${U.esc(f.src)}</span></div>`).join("")}</div>
      </div>
    </div>`;
};

/* ---------------- Section 2 ---------------- */
const How = {};
How.render = () => {
  const stages = C.STAGES();
  const counts = U.primary().meta.class_counts;
  const sampleFor = code => D.samples.lesions.find(l => l.label === code);
  U.$("#s-how").innerHTML = `
    <div class="wrap">
      <div class="section-head">
        <span class="eyebrow">How it works</span>
        <h2>Eight steps from photo to finding</h2>
        <p>The same photos go through every step in this order. Select a step to see what happens, how, why, and what it produces.</p>
      </div>
      <div class="stepper" role="group" aria-label="Pipeline stages">${stages.map((s, i) => `
        <button class="step-btn" type="button" aria-pressed="${i === 0}" data-i="${i}"><span class="k">STEP ${i + 1}</span><b>${s.k}</b></button>`).join("")}</div>
      <div class="panel" id="stage-panel" style="margin-top:12px" aria-live="polite"></div>

      <div class="block">
        <h2>The seven diagnoses</h2>
        <p>The model chooses between these seven lesion types. Counts are the number of photos of each type in the validation set,
          which shows how unbalanced real data is.</p>
        <div class="classes">${D.labels.map(code => {
          const d = C.DIAGNOSES[code] || { kind: "benign", tag: "", text: U.label(code) };
          const s = sampleFor(code);
          return `<div class="cls">
            <div class="row" style="justify-content:space-between"><span class="code">${code}</span><span class="tag ${d.kind}">${d.tag}</span></div>
            <b>${U.esc(U.label(code))}</b><p class="small muted">${d.text}</p>
            <span class="small faint num">${U.int(counts[code] || 0)} photos${s ? ` · example in the Image Lab: ${s.id}` : ""}</span></div>`;
        }).join("")}</div>
      </div>

      <div class="block">
        <h2>How does the AI decide?</h2>
        <p>The model is <b>EfficientNet-B0</b>, a convolutional neural network: a stack of filters that learned from thousands
          of labelled photos which visual patterns go with which diagnosis. It does not reason like a dermatologist; it matches
          patterns. That is exactly why changes in photo quality can change its answer.</p>
        <div class="nn">${C.NN_STEPS.map(([k, t]) => `<div class="nn-step"><span class="k">${k}</span><p class="small">${t}</p></div>`).join("")}</div>
        <p class="small muted">"EfficientNet" refers to a family of networks designed to get high accuracy from few parameters by
          scaling depth, width and image size together. This one is small enough to run on an ordinary laptop CPU.</p>
      </div>
    </div>`;
  const show = i => {
    const s = stages[i];
    U.$("#stage-panel").innerHTML = `
      <div class="stack"><h3>Step ${i + 1}: ${s.k}</h3>
      <div class="step-detail">
        <div><h4>What happens?</h4><p>${s.what}</p></div>
        <div><h4>How does it work?</h4><p>${s.how}</p></div>
        <div><h4>Why is it necessary?</h4><p>${s.why}</p></div>
        <div><h4>What does it produce?</h4><p>${s.out}</p></div>
        <div class="cmd"><span class="label">Command</span><code>${U.esc(s.cmd)}</code></div>
      </div></div>`;
  };
  U.$$(".step-btn").forEach(b => b.addEventListener("click", () => {
    U.pressed(b.parentElement, b);
    show(+b.dataset.i);
  }));
  show(0);
};

window.Overview = Overview;
window.How = How;
