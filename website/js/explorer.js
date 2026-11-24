/* Section 4: AI Prediction Explorer — one lesion across every condition, plus confidence vs accuracy. */
"use strict";

const Explorer = {};
Explorer.state = { lesion: null, cond: "original", filter: "all" };

Explorer.render = () => {
  Explorer.state.lesion = Explorer.state.lesion || D.samples.lesions[0].id;
  const r = U.primary(), key = D.primary, o = U.orig(key);
  const rel = r.reliability.original || [];
  const hi = rel.filter(b => b.confidence >= 0.5);
  const hiN = hi.reduce((a, b) => a + b.count, 0);
  const hiAcc = hiN ? hi.reduce((a, b) => a + b.accuracy * b.count, 0) / hiN : null;
  const hiConf = hiN ? hi.reduce((a, b) => a + b.confidence * b.count, 0) / hiN : null;
  const deg = r.conditions.filter(c => U.catalog[c.name] && U.catalog[c.name].group === "degradation");
  const inverted = deg.filter(c => c.conf_bad != null && c.conf_ok != null && c.conf_bad > c.conf_ok);

  U.$("#s-explorer").innerHTML = `
    <div class="wrap">
      <div class="section-head">
        <span class="eyebrow">AI prediction explorer</span>
        <h2>One lesion, ${U.int(D.catalog.length)} versions, ${U.int(D.catalog.length)} answers</h2>
        <p>The AI gives a probability for each of the seven diagnoses; its answer is the highest one. Pick a lesion, then any
          version of it, to see the full set of probabilities that was saved during the experiment.</p>
      </div>
      <div class="row" id="ex-lesions" style="margin-bottom:14px">${D.samples.lesions.map(l => `
        <button class="chip" type="button" aria-pressed="false" data-id="${l.id}">${l.id} · ${l.label}</button>`).join("")}</div>
      <div class="explorer">
        <div class="panel stack" id="ex-main"></div>
        <div class="stack">
          <div class="row" id="ex-filter">${[["all", "All"], ...Object.entries(U.GROUPS).filter(([k]) => k !== "original").map(([k, g]) => [k, g.name])]
            .map(([k, n], i) => `<button class="chip" type="button" aria-pressed="${i === 0}" data-g="${k}">${k === "all" ? n : `<i class="dot" style="background:${U.GROUPS[k].color}"></i>${n}`}</button>`).join("")}</div>
          <div class="conds-grid" id="ex-grid"></div>
          <p class="small muted" id="ex-summary"></p>
        </div>
      </div>

      <div class="block">
        <h2>Confidence is not the same as being right</h2>
        <p>"Confidence" is the probability the model gives its own answer. It is a score the model produces, not a measured
          chance of being correct. A model can be confidently wrong, or correct while sounding unsure.</p>
        <div class="split">
          <div class="qchart">
            <h3>When the model reports a confidence, how often is it right?</h3>
            <div id="ex-reliability"></div>
            <div class="answer">${hiN ? `On original photos, for the <b>${U.int(hiN)}</b> answers given with at least 50% confidence the model's
              average confidence was <b>${U.pct(hiConf)}</b>, but it was actually right <b>${U.pct(hiAcc)}</b> of the time.` : ""}
              Overall it is <b>${o.conf < o.acc ? "under-confident" : "over-confident"}</b>: average confidence ${U.pct(o.conf)}
              versus accuracy ${U.pct(o.acc)} (calibration error ${U.f3(o.ece)}).</div>
            ${U.source(key, "details", "reliability_bins (original)")}
          </div>
          <div class="qchart">
            <h3>Are wrong answers less confident than right ones?</h3>
            <div id="ex-confgap"></div>
            <div class="answer">On original photos, right answers averaged <b>${U.pct(o.conf_ok)}</b> confidence and wrong answers
              <b>${U.pct(o.conf_bad)}</b>. Under ${inverted.length} of ${deg.length} damage conditions, wrong answers were on average
              <b>more</b> confident than right ones, so confidence alone cannot warn you that a damaged photo misled the model.</div>
            ${U.source(key, "summary", "mean_conf_correct, mean_conf_wrong")}
          </div>
        </div>
        <div class="callout"><b>Why can a model be confident and still wrong?</b>
          <p class="small">It only knows the patterns in its training photos. A blurred melanoma loses the fine irregular texture
            that marks it as cancer and starts to resemble a smooth, ordinary mole, a pattern the model has seen thousands of times.
            It then matches that pattern with conviction. Confidence measures how familiar the pattern looks, not whether the
            diagnosis is true.</p></div>
      </div>
    </div>`;

  U.$$("#ex-lesions .chip").forEach(b => b.addEventListener("click", () => { Explorer.state.lesion = b.dataset.id; Explorer.update(); }));
  U.$$("#ex-filter .chip").forEach(b => b.addEventListener("click", () => {
    Explorer.state.filter = b.dataset.g; U.pressed(U.$("#ex-filter"), b); Explorer.update();
  }));
  Explorer.reliability(key);
  Explorer.confGap(key);
  Explorer.update();
};

Explorer.update = () => {
  const st = Explorer.state, les = U.lesion(st.lesion);
  U.$$("#ex-lesions .chip").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.id === st.lesion)));
  const names = D.catalog.map(c => c.name).filter(n => st.filter === "all" || n === "original" || U.catalog[n].group === st.filter);
  if (!names.includes(st.cond)) st.cond = "original";
  const grid = U.$("#ex-grid");
  grid.innerHTML = names.map(n => {
    const rec = les.conditions[n];
    return `<button class="cond-card" type="button" aria-pressed="${n === st.cond}" data-c="${n}">
      <span class="groupbar" style="background:${U.groupColor(n)}"></span>
      <img src="${rec.img}" alt="${U.esc(U.title(n))}" loading="lazy">
      <span class="mark ${rec.correct ? "ok" : "bad"}">${rec.correct ? "✓" : "✗"}</span>
      <span class="meta"><b>${U.esc(U.shortTitle(n))}</b><span>${rec.pred} · ${U.pct(rec.conf)}</span></span></button>`;
  }).join("");
  U.$$(".cond-card", grid).forEach(b => b.addEventListener("click", () => { st.cond = b.dataset.c; Explorer.update(); }));
  const all = D.catalog.map(c => les.conditions[c.name]);
  const right = all.filter(x => x.correct).length;
  U.$("#ex-summary").textContent = `${les.id} was classified correctly in ${right} of ${all.length} versions. ✓/✗ compare the AI's answer with the true diagnosis (${les.label}).`;

  const rec = les.conditions[st.cond], o = les.conditions.original;
  U.$("#ex-main").innerHTML = `
    <div class="row" style="justify-content:space-between"><h3>${U.esc(U.title(st.cond))}</h3>${U.verdict(rec)}</div>
    <div class="frame"><div class="img"><img src="${rec.img}" alt="${U.esc(U.title(st.cond))} of ${les.id}"></div></div>
    <table class="t">
      <tr><th>Ground truth</th><td><b>${les.label}</b> · ${U.esc(U.label(les.label))}</td></tr>
      <tr><th>Highest predicted class</th><td><b>${rec.pred}</b> · ${U.esc(U.label(rec.pred))}</td></tr>
      <tr><th>Confidence</th><td class="num">${U.pct(rec.conf, 1)} <span class="faint small">(gap to 2nd choice ${U.pct(rec.margin, 1)})</span></td></tr>
      ${st.cond !== "original" ? `<tr><th>Versus original</th><td>${rec.pred === o.pred ? "same answer" : `changed from <b>${o.pred}</b>`} ·
        confidence ${U.pct(o.conf)} → ${U.pct(rec.conf)}</td></tr>` : ""}
    </table>
    ${U.probBars(rec, les.label)}${U.probLegend()}
    <p class="small faint">From <code>${U.esc(D.samples.source)}</code>, row ${les.id} / ${st.cond}.</p>`;
};

Explorer.reliability = key => {
  const bins = U.results(key).reliability.original || [];
  const W = 360, H = 260, L = 44, R = 12, T = 12, B = 40;
  const x = U.linear(0, 1, L, W - R), y = U.linear(0, 1, H - B, T);
  let s = "";
  [0, 0.25, 0.5, 0.75, 1].forEach(g => {
    s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - B}"/><line class="grid" x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}"/>`;
    s += `<text x="${x(g)}" y="${H - B + 14}" text-anchor="middle">${U.pct(g)}</text><text x="${L - 6}" y="${y(g) + 4}" text-anchor="end">${U.pct(g)}</text>`;
  });
  s += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" class="base"/>`;
  s += `<text x="${x(0.78)}" y="${y(0.66)}" text-anchor="middle" transform="rotate(-34 ${x(0.78)} ${y(0.66)})" style="font-size:10px">perfectly honest</text>`;
  const pts = bins.map(b => `${x(b.confidence)},${y(b.accuracy)}`).join(" ");
  s += `<polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
  bins.forEach((b, i) => {
    const r = Math.max(4, Math.min(11, Math.sqrt(b.count) * 1.1));
    s += `<circle cx="${x(b.confidence)}" cy="${y(b.accuracy)}" r="${r}" fill="var(--accent)" stroke="#fff" stroke-width="2" data-i="${i}" tabindex="0"/>`;
  });
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">model's confidence</text>`;
  s += `<text x="12" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(T + H - B) / 2})">actually correct</text>`;
  const el = U.$("#ex-reliability");
  el.innerHTML = U.svg(W, H, "Reliability diagram for original photos", s);
  U.$$("circle", el).forEach(c => {
    const b = bins[+c.dataset.i];
    U.hover(c, `<b>${b.count} answers</b> with confidence ${U.pct(b.bin_low)}–${U.pct(b.bin_high)}<br>average confidence ${U.pct(b.confidence)} · actually right ${U.pct(b.accuracy)}`);
  });
};

Explorer.confGap = key => {
  const r = U.results(key);
  const names = ["original", ...["gaussian_blur", "motion_blur", "underexposure", "overexposure", "low_contrast",
    "low_resolution", "jpeg_compression", "gaussian_noise"].map(d => `${d}__moderate`)].filter(n => U.cond(key, n));
  const W = 380, rowH = 22, L = 130, R = 16, T = 10;
  const H = T + names.length * rowH + 30;
  const x = U.linear(0.3, 0.8, L, W - R);
  let s = "";
  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8].forEach(g => {
    s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - 26}"/><text x="${x(g)}" y="${H - 12}" text-anchor="middle">${U.pct(g)}</text>`;
  });
  names.forEach((n, i) => {
    const c = U.cond(key, n), yy = T + i * rowH + 11;
    const clamp = v => Math.max(0.3, Math.min(0.8, v));
    s += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end" style="fill:var(--ink)">${U.esc(n === "original" ? "Original" : U.NAMES[U.catalog[n].degradation])}</text>`;
    s += `<line x1="${x(clamp(c.conf_ok))}" x2="${x(clamp(c.conf_bad))}" y1="${yy}" y2="${yy}" stroke="var(--ink-3)" stroke-width="1.5"/>`;
    s += `<circle cx="${x(clamp(c.conf_ok))}" cy="${yy}" r="5" fill="var(--ok)" stroke="#fff" stroke-width="1.5"/>`;
    s += `<circle cx="${x(clamp(c.conf_bad))}" cy="${yy}" r="5" fill="var(--bad)" stroke="#fff" stroke-width="1.5"/>`;
    s += `<rect x="0" y="${yy - 11}" width="${W}" height="${rowH}" fill="transparent" data-n="${n}"/>`;
  });
  const el = U.$("#ex-confgap");
  el.innerHTML = U.svg(W, H, "Average confidence of right and wrong answers", s) +
    `<div class="legend"><span><i class="dot" style="background:var(--ok)"></i>right answers</span><span><i class="dot" style="background:var(--bad)"></i>wrong answers</span><span>(moderate damage)</span></div>`;
  U.$$("rect[data-n]", el).forEach(rc => {
    const c = U.cond(key, rc.dataset.n);
    U.hover(rc, `<b>${U.esc(U.title(rc.dataset.n))}</b><br>right answers: ${U.pct(c.conf_ok, 1)} average confidence<br>wrong answers: ${U.pct(c.conf_bad, 1)}`);
  });
};

window.Explorer = Explorer;
