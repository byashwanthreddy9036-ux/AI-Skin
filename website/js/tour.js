/* Guided tour: seven full-screen steps for presenting to an audience. Keys: ← → to move, Esc to close. */
"use strict";

const Tour = {};
Tour.i = 0;

Tour.slides = () => {
  const key = D.primary, r = U.primary(), m = r.meta, o = U.orig(key);
  const blur = D.samples.stories.find(s => s.key === "blur") || D.samples.stories[0];
  const les = U.lesion(blur.image_id);
  const bo = les.conditions.original, bt = les.conditions[blur.condition];
  const counts = m.class_counts;
  const maxCount = Math.max(...Object.values(counts));
  const types = [...new Set(D.catalog.filter(c => c.group === "degradation").map(c => c.degradation))];
  const others = r.conditions.filter(c => c.name !== "original");
  const worse = others.filter(c => c.d_hi < 0);
  const enh = others.filter(c => U.catalog[c.name].group === "enhancement");
  const enhBetter = enh.filter(c => c.d_lo > 0);
  const fix = r.restoration.find(f => f.name === "underexposure__moderate+brightness_correction");
  const melO = r.per_class.original.mel.recall, melB = r.per_class.gaussian_blur__mild.mel.recall;

  const bars = D.labels.map(l => `<div class="prob"><b class="mono">${l}</b><div class="bar"><i style="width:${(100 * (counts[l] || 0) / maxCount).toFixed(1)}%;background:var(--accent)"></i></div><span class="pct">${counts[l] || 0}</span></div>`).join("");
  const sevRows = types.map(d => {
    const s = U.cond(key, `${d}__severe`);
    return `<div class="prob"><b style="width:auto">${U.NAMES[d]}</b><div class="bar"><i style="width:${(s.bal * 100).toFixed(1)}%;background:var(--c-deg)"></i></div><span class="pct">${U.f2(s.bal)}</span></div>`;
  }).join("");

  return [
    { t: "The problem", h: "Skin-cancer AI is tested on perfect photos. Real photos are not perfect.",
      p: "Dermatology AI models are trained and tested on high-quality dermoscopic photos. In real use, photos can be blurry, dark or compressed. Does that make the AI less reliable?",
      v: `<div class="compare"><figure class="frame" style="margin:0"><div class="img"><img src="${bo.img}" alt="original"></div><figcaption class="cap"><span>A sharp photo</span></figcaption></figure>
          <figure class="frame" style="margin:0"><div class="img"><img src="${bt.img}" alt="blurred"></div><figcaption class="cap"><span>The same lesion, slightly out of focus</span></figcaption></figure></div>` },
    { t: "The dataset", h: `${U.int(m.n_images)} expert-labelled photos the AI had never seen`,
      p: `Photos from the ISIC 2018 challenge validation set: ${U.int(m.n_lesions)} lesions across seven diagnoses. Most are ordinary moles (nv), which is why we score the AI by its average success on each diagnosis, not overall.`,
      v: `<div class="panel stack"><span class="label">Photos per diagnosis</span><div class="probs">${bars}</div></div>` },
    { t: "Transforming the image", h: `${U.int(m.n_conditions)} versions of every photo`,
      p: `${types.length} kinds of damage (blur, camera shake, dark, bright, washed-out, low resolution, JPEG, noise) at three strengths, plus classic enhancements, and damage followed by a fix. Drag the slider to compare.`,
      v: `<div class="slider" id="tour-slider"><img src="${bo.img}" alt="original"><img class="top" src="${bt.img}" alt="${U.esc(U.title(blur.condition))}"><div class="handle"></div>
          <span class="tagl">Original</span><span class="tagr">${U.esc(U.shortTitle(blur.condition))}</span><input type="range" min="0" max="100" value="50" aria-label="Compare"></div>` },
    { t: "How the AI responds", h: bt.correct ? "The answer can shift" : `${U.label(bo.pred)} becomes ${U.label(bt.pred)}`,
      p: `Lesion ${les.id} is a ${U.label(les.label).toLowerCase()}. On the original photo the AI said <b>${bo.pred}</b> (${U.pct(bo.conf)}). After ${U.title(blur.condition)} it said <b>${bt.pred}</b> (${U.pct(bt.conf)}).`,
      v: `<div class="split"><div class="panel stack"><h4>Original ${U.verdict(bo)}</h4>${U.probBars(bo, les.label)}</div>
          <div class="panel stack"><h4>${U.esc(U.shortTitle(blur.condition))} ${U.verdict(bt)}</h4>${U.probBars(bt, les.label)}</div></div>${U.probLegend()}` },
    { t: "What we measure", h: "Balanced accuracy, with honest error bars",
      p: `For every version we compare the AI's answers with the true diagnoses. On the original photos balanced accuracy is ${U.f3(o.bal)} (95% range ${U.f3(o.bal_lo)}–${U.f3(o.bal_hi)}). A change only counts if its range of plausible values excludes "no change".`,
      v: `<div class="panel stack" style="text-align:center"><span class="label">Original photos</span><span class="big-num">${U.f3(o.bal)}</span><span class="muted">balanced accuracy · ${U.int(m.bootstrap)} bootstrap resamples of whole lesions</span></div>` },
    { t: "What the results reveal", h: `${worse.length} of ${others.length} changes made the AI worse`,
      p: `Every damage type hurt more as it got stronger. ${enhBetter.length === 0 ? "None" : enhBetter.length} of ${enh.length} enhancements improved good photos.${fix ? ` A fix helps only when it reverses the damage: brightness correction recovered ${U.pct(fix.recovery)} of the accuracy lost to poor lighting.` : ""}`,
      v: `<div class="panel stack"><span class="label">Balanced accuracy at severe damage (original ${U.f2(o.bal)})</span><div class="probs">${sevRows}</div></div>` },
    { t: "Why it matters", h: `Mild blur: melanoma detection ${U.pct(melO)} → ${U.pct(melB)}`,
      p: "A photo slightly out of focus made the AI miss most melanomas, often calling them harmless moles, and its confidence barely changed. Image quality checks should come before any AI skin assessment, and AI tools should be tested on realistic photos, not only perfect ones.",
      v: `<div class="panel stack"><span class="big-num" style="color:var(--bad)">${U.pct(melB)}</span><span class="muted">of melanomas detected after mild blur (${U.pct(melO)} on original photos)</span>
          <div class="notice small">Research experiment, not a diagnostic tool. ${U.int(m.n_images)} photos, one model; see About for limitations.</div></div>` },
  ];
};

Tour.open = (i = 0) => {
  Tour.i = i;
  U.$("#tour").hidden = false;
  document.body.style.overflow = "hidden";
  Tour.draw();
  U.$("#tour-next").focus();
};
Tour.close = () => {
  U.$("#tour").hidden = true;
  document.body.style.overflow = "";
};
Tour.go = d => {
  const n = Tour.slides().length;
  Tour.i = Math.max(0, Math.min(n - 1, Tour.i + d));
  Tour.draw();
};
Tour.draw = () => {
  const slides = Tour.slides(), s = slides[Tour.i];
  U.$("#tour-progress").innerHTML = slides.map((_, i) => `<i class="${i <= Tour.i ? "on" : ""}"></i>`).join("");
  U.$("#tour-count").textContent = `${Tour.i + 1} / ${slides.length} · ${s.t}`;
  U.$("#tour-body").innerHTML = `<div class="tour-slide"><div class="text"><span class="eyebrow">${Tour.i + 1} · ${s.t}</span><h2>${s.h}</h2><p>${s.p}</p></div><div>${s.v}</div></div>`;
  U.$("#tour-prev").disabled = Tour.i === 0;
  U.$("#tour-next").textContent = Tour.i === slides.length - 1 ? "Finish" : "Next →";
  const sl = U.$("#tour-slider");
  if (sl) {
    const input = U.$("input", sl);
    const set = v => { U.$(".top", sl).style.clipPath = `inset(0 0 0 ${v}%)`; U.$(".handle", sl).style.left = `${v}%`; };
    input.addEventListener("input", () => set(input.value));
    set(50);
  }
};
Tour.init = () => {
  U.$("#tour-prev").addEventListener("click", () => Tour.go(-1));
  U.$("#tour-next").addEventListener("click", () => {
    if (Tour.i === Tour.slides().length - 1) Tour.close(); else Tour.go(1);
  });
  U.$("#tour-close").addEventListener("click", Tour.close);
  document.addEventListener("keydown", e => {
    if (U.$("#tour").hidden) return;
    if (e.key === "ArrowRight") Tour.go(1);
    else if (e.key === "ArrowLeft") Tour.go(-1);
    else if (e.key === "Escape") Tour.close();
  });
  document.addEventListener("click", e => {
    const t = e.target.closest("[data-tour]");
    if (t) { e.preventDefault(); Tour.open(0); }
  });
};

window.Tour = Tour;
