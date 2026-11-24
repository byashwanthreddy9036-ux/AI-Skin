/* Shared helpers: data access, formatting, DOM, tooltip and small SVG chart primitives.
   Everything reads window.SITE_DATA (written by `skin-research build-site`). */
"use strict";

const D = window.SITE_DATA;
const U = {};

U.$ = (sel, root = document) => root.querySelector(sel);
U.$$ = (sel, root = document) => [...root.querySelectorAll(sel)];
U.esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
U.h = (tag, attrs = {}, html = "") => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    el.setAttribute(k, v === true ? "" : v);
  }
  if (html) el.innerHTML = html;
  return el;
};

/* ---------- formatting ---------- */
U.f2 = x => (x == null ? "–" : Number(x).toFixed(2));
U.f3 = x => (x == null ? "–" : Number(x).toFixed(3));
U.pct = (x, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
U.signed = (x, d = 3) => (x == null ? "–" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`);
U.int = x => (x == null ? "–" : Number(x).toLocaleString("en"));

/* ---------- experiments ---------- */
U.exp = key => D.experiments[key];
U.results = key => (D.experiments[key] && D.experiments[key].results) || null;
U.primary = () => U.results(D.primary);
U.isComplete = key => D.experiments[key] && D.experiments[key].status.state === "complete";
U.cond = (key, name) => {
  const r = U.results(key);
  return r ? r.conditions.find(c => c.name === name) : null;
};
U.orig = key => U.cond(key, "original");

/* ---------- conditions ---------- */
U.catalog = Object.fromEntries(D.catalog.map(c => [c.name, c]));
U.GROUPS = {
  original: { name: "Original", color: "var(--c-orig)" },
  degradation: { name: "Damaged", color: "var(--c-deg)" },
  enhancement: { name: "Enhanced original", color: "var(--c-enh)" },
  degrade_enhance: { name: "Damaged, then fixed", color: "var(--c-fix)" },
};
U.NAMES = {
  gaussian_blur: "Gaussian blur", motion_blur: "Camera shake", underexposure: "Underexposure",
  overexposure: "Overexposure", low_contrast: "Low contrast", low_resolution: "Low resolution",
  jpeg_compression: "JPEG compression", gaussian_noise: "Sensor noise",
  brightness_correction: "Brightness correction", contrast_stretch: "Contrast stretch", clahe: "CLAHE",
  sharpen: "Sharpening", denoise: "Denoising", deblur: "Deblurring",
};
U.title = name => {
  const c = U.catalog[name];
  if (!c || name === "original") return "Original photo";
  if (c.group === "degradation") return `${U.NAMES[c.degradation]} · ${c.severity}`;
  if (c.group === "enhancement") return U.NAMES[c.enhancement];
  return `${U.NAMES[c.degradation]} (${c.severity}) → ${U.NAMES[c.enhancement]}`;
};
U.shortTitle = name => {
  const c = U.catalog[name];
  if (!c || name === "original") return "Original";
  if (c.group === "degradation") return `${U.NAMES[c.degradation]} (${c.severity})`;
  if (c.group === "enhancement") return U.NAMES[c.enhancement];
  return `${U.NAMES[c.degradation]} → ${U.NAMES[c.enhancement]}`;
};
U.params = name => {
  const c = U.catalog[name];
  if (!c) return "";
  return Object.entries({ ...(c.degradation_params || {}), ...(c.enhancement_params || {}) })
    .map(([k, v]) => `${k} = ${v}`).join(", ");
};
U.groupColor = name => U.GROUPS[(U.catalog[name] || {}).group || "original"].color;
U.label = code => D.label_names[code] || code;
U.labelIdx = code => D.labels.indexOf(code);

/* ---------- samples ---------- */
U.lesion = id => D.samples.lesions.find(l => l.id === id);
U.img = (lesionId, cond) => {
  const l = U.lesion(lesionId);
  return l && l.conditions[cond] ? l.conditions[cond].img : null;
};

/* ---------- tooltip ---------- */
const tip = () => U.$("#tip");
U.showTip = (evt, html) => {
  const t = tip();
  t.innerHTML = html; t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 10);
  const y = Math.max(10, evt.clientY - t.offsetHeight - 14);
  t.style.left = `${Math.max(10, x)}px`; t.style.top = `${y}px`;
};
U.hideTip = () => { tip().hidden = true; };
U.hover = (el, html) => {
  el.addEventListener("pointerenter", e => U.showTip(e, html));
  el.addEventListener("pointermove", e => U.showTip(e, html));
  el.addEventListener("pointerleave", U.hideTip);
  el.addEventListener("focus", () => {
    const r = el.getBoundingClientRect();
    U.showTip({ clientX: r.left + r.width / 2, clientY: r.top }, html);
  });
  el.addEventListener("blur", U.hideTip);
};

/* ---------- toggle groups ---------- */
U.pressed = (container, active) => U.$$("[aria-pressed]", container).forEach(b => b.setAttribute("aria-pressed", String(b === active)));

/* ---------- SVG ---------- */
U.svg = (w, h, label, inner) =>
  `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${U.esc(label)}" preserveAspectRatio="xMinYMin meet">${inner}</svg>`;
U.linear = (d0, d1, r0, r1) => v => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

/* Colour cell for a value in [0,1] on the sequential blue ramp (text colour flips for contrast). */
U.seqCell = v => {
  if (v == null) return `style="background:var(--surface-2);color:var(--ink-3)"`;
  const p = Math.round(6 + v * 88);
  return `style="background:color-mix(in oklab, var(--c-seq) ${p}%, #ffffff);color:${v > 0.55 ? "#fff" : "var(--ink)"}"`;
};

/* Probability bars for one prediction record. */
U.probBars = (rec, truth) => {
  const order = D.labels.map((_, i) => i).sort((a, b) => rec.probs[b] - rec.probs[a]);
  const top = order[0];
  const ti = U.labelIdx(truth);
  return `<div class="probs">${order.map(i => `
      <div class="prob ${i === ti ? "truth" : ""} ${i === top && i !== ti ? "top" : ""}" title="${U.esc(U.label(D.labels[i]))}">
        <b class="mono">${D.labels[i]}</b>
        <div class="bar"><i style="width:${(rec.probs[i] * 100).toFixed(1)}%"></i></div>
        <span class="pct">${(rec.probs[i] * 100).toFixed(1)}%</span>
      </div>`).join("")}</div>`;
};
U.probLegend = () => `<div class="prob-legend">
    <span><i style="background:var(--accent)"></i>true diagnosis</span>
    <span><i style="background:var(--ink)"></i>model's top answer (when wrong)</span>
    <span><i style="background:var(--ink-3)"></i>other diagnoses</span></div>`;
U.verdict = rec => rec.correct
  ? `<span class="verdict ok">✓ correct</span>`
  : `<span class="verdict bad">✗ wrong</span>`;

/* Source footnote for a chart. */
U.source = (key, file, cols) => {
  const r = U.results(key);
  const path = r ? r.sources[file] : file;
  return `<div class="src">Source: <code>${U.esc(path)}</code>${cols ? ` · ${U.esc(cols)}` : ""} · experiment <code>${U.esc(r ? r.meta.experiment : key)}</code></div>`;
};

window.U = U;
