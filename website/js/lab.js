/* Section 3: Interactive Image Lab (curated gallery + original vs transformed comparison). */
"use strict";

const Lab = {};

/* Transformation menu: label → condition family. Severity applies to damage types. */
Lab.MENU = [
  { group: "Damage", items: [
    ["gaussian_blur", "Gaussian blur"], ["motion_blur", "Camera shake"], ["underexposure", "Underexposure"],
    ["overexposure", "Overexposure"], ["low_contrast", "Low contrast"], ["gaussian_noise", "Sensor noise"],
    ["jpeg_compression", "JPEG compression"], ["low_resolution", "Low resolution"]] },
  { group: "Enhance a good photo", items: [
    ["enhance__sharpen", "Sharpening"], ["enhance__denoise", "Denoising"], ["enhance__clahe", "CLAHE"],
    ["enhance__brightness_correction", "Brightness correction"], ["enhance__contrast_stretch", "Contrast stretch"]] },
  { group: "Damage, then fix", items: D.catalog.filter(c => c.group === "degrade_enhance")
      .map(c => [c.name, `${U.NAMES[c.degradation]} → ${U.NAMES[c.enhancement]}`]) },
];
Lab.SEVERITIES = ["mild", "moderate", "severe"];
Lab.state = { lesion: null, family: "gaussian_blur", severity: "mild", view: "side" };

Lab.condName = () => {
  const f = Lab.state.family;
  return U.catalog[`${f}__${Lab.state.severity}`] ? `${f}__${Lab.state.severity}` : f;
};
Lab.familyOf = cond => {
  const c = U.catalog[cond];
  if (!c || cond === "original") return { family: "gaussian_blur", severity: "mild" };
  if (c.group === "degradation") return { family: c.degradation, severity: c.severity };
  return { family: cond, severity: Lab.state.severity };
};

Lab.render = () => {
  const stories = D.samples.stories;
  Lab.state.lesion = Lab.state.lesion || (stories.find(s => s.key === "blur") || stories[0]).image_id;
  const missing = D.samples.missing_stories || [];
  U.$("#s-lab").innerHTML = `
    <div class="wrap">
      <div class="section-head">
        <span class="eyebrow">Interactive image lab</span>
        <h2>Change the photo, watch the AI's answer</h2>
        <p>Every image here is a real photo from the ISIC 2018 validation set, or the exact transformed copy the AI
          was shown during the experiment. Predictions are the saved experiment results, not recomputed in your browser.</p>
      </div>

      <div class="block" style="margin-top:0">
        <h3>Curated cases</h3>
        <p class="small muted">Chosen automatically by fixed rules from the saved predictions (melanoma preferred, one photo per
          lesion). Select a case to load it into the lab below.${missing.length ? ` No qualifying photo existed for: ${missing.join(", ")}.` : ""}</p>
        <div class="gallery" id="gallery">${stories.map((s, i) => Lab.caseCard(s, i)).join("")}</div>
      </div>

      <div class="block">
        <div class="lab">
          <div class="lab-controls panel">
            <div class="stack"><span class="label">1 · Choose a lesion</span>
              <div class="lesions" id="lesions">${D.samples.lesions.map(l => `
                <button class="lesion-btn" type="button" aria-pressed="false" data-id="${l.id}" aria-label="${l.id}, ${U.esc(U.label(l.label))}">
                  <img src="${l.thumb}" alt="" loading="lazy"><span>${l.label}</span></button>`).join("")}</div>
              <span class="small muted" id="lesion-caption"></span>
            </div>
            <div class="stack"><span class="label">2 · Choose a change</span>
              <div class="tgroups" id="tgroups">${Lab.MENU.map(g => `
                <div class="stack" style="gap:6px"><span class="small faint">${g.group}</span>
                  <div class="row">${g.items.map(([f, n]) => `<button class="chip" type="button" aria-pressed="false" data-f="${f}">${U.esc(n)}</button>`).join("")}</div></div>`).join("")}
              </div>
            </div>
            <div class="stack"><span class="label">3 · Strength</span>
              <div class="seg" id="severity" role="group" aria-label="Strength">${Lab.SEVERITIES.map(s =>
                `<button type="button" aria-pressed="false" data-s="${s}">${s}</button>`).join("")}</div>
              <span class="small faint" id="severity-note"></span>
            </div>
          </div>

          <div class="viewer">
            <div class="row" style="justify-content:space-between">
              <h3 id="lab-title"></h3>
              <div class="seg" id="viewmode" role="group" aria-label="View">
                <button type="button" aria-pressed="true" data-v="side">Side by side</button>
                <button type="button" aria-pressed="false" data-v="slider">Before / after slider</button>
              </div>
            </div>
            <div id="lab-images"></div>
            <div class="cond-note" id="lab-note"></div>
            <div class="split">
              <div class="panel stack"><h4>What the AI said</h4><div id="lab-preds"></div></div>
              <div class="panel stack"><h4>Image quality</h4><div id="lab-quality"></div></div>
            </div>
            <div class="split">
              <div class="panel stack"><h4>Original: all 7 probabilities</h4><div id="lab-prob-o"></div></div>
              <div class="panel stack"><h4 id="lab-prob-t-title">Transformed: all 7 probabilities</h4><div id="lab-prob-t"></div></div>
            </div>
            ${U.probLegend()}
          </div>
        </div>
      </div>
    </div>`;

  U.$$("#gallery .case").forEach(b => b.addEventListener("click", () => {
    const s = stories[+b.dataset.i];
    Lab.state.lesion = s.image_id;
    Object.assign(Lab.state, Lab.familyOf(s.condition === "original" ? "gaussian_blur__mild" : s.condition));
    U.pressed(U.$("#gallery"), b);
    Lab.update();
    U.$("#lab-title").scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  U.$$("#lesions .lesion-btn").forEach(b => b.addEventListener("click", () => { Lab.state.lesion = b.dataset.id; Lab.update(); }));
  U.$$("#tgroups .chip").forEach(b => b.addEventListener("click", () => { Lab.state.family = b.dataset.f; Lab.update(); }));
  U.$$("#severity button").forEach(b => b.addEventListener("click", () => { Lab.state.severity = b.dataset.s; Lab.update(); }));
  U.$$("#viewmode button").forEach(b => b.addEventListener("click", () => {
    Lab.state.view = b.dataset.v; U.pressed(U.$("#viewmode"), b); Lab.update();
  }));
  Lab.update();
};

Lab.caseCard = (s, i) => {
  const l = U.lesion(s.image_id);
  const o = l.conditions.original, t = l.conditions[s.condition];
  const shown = s.condition === "original" ? null : t;
  return `<button class="case" type="button" aria-pressed="false" data-i="${i}">
    <div class="pair"><img src="${o.img}" alt="Original ${s.image_id}" loading="lazy">
      <img src="${(shown || o).img}" alt="${U.esc(U.title(s.condition))} ${s.image_id}" loading="lazy"></div>
    <div class="body"><b>${U.esc(s.title)}</b>
      <span class="small mono faint">${s.image_id} · true: ${s.label}</span>
      <span class="small">${U.esc(U.title(s.condition))}</span>
      <span class="small">Original → <b>${o.pred}</b> ${U.pct(o.conf)}${shown ? ` · then → <b>${shown.pred}</b> ${U.pct(shown.conf)} ${shown.correct ? "✓" : "✗"}` : " ✓"}</span>
    </div></button>`;
};

Lab.update = () => {
  const st = Lab.state;
  const les = U.lesion(st.lesion);
  const cond = Lab.condName();
  const o = les.conditions.original, t = les.conditions[cond];
  const isDamage = !!U.catalog[`${st.family}__mild`];

  U.$$("#lesions .lesion-btn").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.id === st.lesion)));
  U.$$("#tgroups .chip").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.f === st.family)));
  U.$$("#severity button").forEach(b => {
    b.disabled = !isDamage;
    b.setAttribute("aria-pressed", String(isDamage && b.dataset.s === st.severity));
  });
  U.$("#severity-note").textContent = isDamage ? "" : (U.catalog[cond].group === "degrade_enhance"
    ? `Fixes were tested after ${U.catalog[cond].severity} damage only.` : "Enhancements have one setting.");
  U.$("#lesion-caption").innerHTML = `<b>${les.id}</b> · true diagnosis: <b>${U.esc(U.label(les.label))}</b> (${les.label})`;
  U.$("#lab-title").textContent = U.title(cond);

  const changed = t.pred !== o.pred;
  const tCap = `<span>${U.esc(U.shortTitle(cond))}</span><span>AI: <b>${t.pred}</b> ${U.pct(t.conf)} ${t.correct ? "✓" : "✗"}</span>`;
  const oCap = `<span>Original</span><span>AI: <b>${o.pred}</b> ${U.pct(o.conf)} ${o.correct ? "✓" : "✗"}</span>`;
  U.$("#lab-images").innerHTML = st.view === "side"
    ? `<div class="compare">
        <figure class="frame" style="margin:0"><div class="img"><img src="${o.img}" alt="Original photo ${les.id}"></div><figcaption class="cap">${oCap}</figcaption></figure>
        <figure class="frame" style="margin:0"><div class="img"><img src="${t.img}" alt="${U.esc(U.title(cond))} of ${les.id}"></div><figcaption class="cap">${tCap}</figcaption></figure>
      </div>`
    : `<div class="slider" id="slider">
        <img src="${o.img}" alt="Original photo ${les.id}">
        <img class="top" src="${t.img}" alt="${U.esc(U.title(cond))} of ${les.id}">
        <div class="handle"></div><span class="tagl">Original</span><span class="tagr">${U.esc(U.shortTitle(cond))}</span>
        <input type="range" min="0" max="100" value="50" aria-label="Drag to compare original and transformed photo">
      </div>
      <div class="row small" style="justify-content:space-between"><span>${oCap}</span><span>${tCap}</span></div>`;
  if (st.view === "slider") {
    const sl = U.$("#slider"), input = U.$("input", sl);
    const set = v => { U.$(".top", sl).style.clipPath = `inset(0 0 0 ${v}%)`; U.$(".handle", sl).style.left = `${v}%`; };
    input.addEventListener("input", () => set(input.value));
    set(50);
  }

  const c = U.catalog[cond];
  const kind = c.group === "enhancement" ? c.enhancement : c.degradation;
  const fixText = c.group === "degrade_enhance" ? `<p class="small"><b>Then fixed with ${U.NAMES[c.enhancement]}:</b> ${C.TRANSFORMS[c.enhancement]}</p>` : "";
  const agg = U.cond(D.primary, cond);
  U.$("#lab-note").innerHTML = `<p><b>What this simulates:</b> ${C.TRANSFORMS[kind] || ""}</p>${fixText}
    <p class="small muted">Settings: <span class="mono">${U.esc(U.params(cond))}</span>${agg ? ` · Across all ${U.int(U.primary().meta.n_images)} photos this
      change flipped the AI's answer for <b>${U.pct(agg.flip)}</b> of them and gave balanced accuracy <b>${U.f3(agg.bal)}</b>
      (original ${U.f3(U.orig(D.primary).bal)}).` : ""}</p>`;

  const ti = U.labelIdx(les.label);
  U.$("#lab-preds").innerHTML = `<table class="t">
    <tr><th>Ground-truth diagnosis</th><td colspan="2"><b>${les.label}</b> · ${U.esc(U.label(les.label))}</td></tr>
    <tr><th></th><th>Original</th><th>Transformed</th></tr>
    <tr><th>AI prediction</th><td><b>${o.pred}</b></td><td><b>${t.pred}</b></td></tr>
    <tr><th>Confidence</th><td class="num">${U.pct(o.conf, 1)}</td><td class="num">${U.pct(t.conf, 1)}</td></tr>
    <tr><th>Probability of true diagnosis</th><td class="num">${U.pct(o.probs[ti], 1)}</td><td class="num">${U.pct(t.probs[ti], 1)}</td></tr>
    <tr><th>Result</th><td>${U.verdict(o)}</td><td>${U.verdict(t)}</td></tr>
    </table>
    <p>${changed ? `<span class="verdict changed">Prediction changed</span> from <b>${o.pred}</b> to <b>${t.pred}</b>.`
      : `<span class="verdict same">Prediction unchanged</span> (${o.pred}).`}</p>`;

  const q = (k, digits) => [o.quality[k], t.quality[k]].map(v => v == null ? "–" : Number(v).toFixed(digits));
  const rows = [["Brightness (0 = black, 1 = white)", "brightness", 3], ["Contrast (spread of tones)", "contrast", 3],
    ["Blur score (higher = more edges)", "blur", 0], ["Noise estimate", "noise", 2]];
  U.$("#lab-quality").innerHTML = `<table class="t"><tr><th>Measure</th><th class="n">Original</th><th class="n">Transformed</th></tr>
    ${rows.map(([n, k, d]) => { const [a, b] = q(k, d); return `<tr><td>${n}</td><td class="n">${a}</td><td class="n">${b}</td></tr>`; }).join("")}
    <tr><td>Quality warnings</td><td>${o.flags ? U.esc(o.flags.replaceAll(";", ", ")) : "none"}</td><td>${t.flags ? U.esc(t.flags.replaceAll(";", ", ")) : "none"}</td></tr></table>
    <p class="small muted">The blur score counts edges, so noise and JPEG blocks can raise it even though the photo got worse. Quality
      warnings use simple thresholds and are not clinical criteria.</p>`;

  U.$("#lab-prob-o").innerHTML = U.probBars(o, les.label);
  U.$("#lab-prob-t").innerHTML = U.probBars(t, les.label);
  U.$("#lab-prob-t-title").textContent = `${U.shortTitle(cond)}: all 7 probabilities`;
};

window.Lab = Lab;
