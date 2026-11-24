/* Section 5: Research results. Every chart reads the selected experiment's exported tables. */
"use strict";

const Results = {};
Results.key = null;

Results.statusText = st => ({
  complete: "complete",
  running: st.total ? `running, ${U.pct((st.done || 0) / st.total)} of predictions done${st.stale ? " (no update for hours; may be paused)" : ""}` : `running (${st.stage || "starting"})`,
  not_started: "not started yet",
  not_prepared: "dataset not prepared yet",
  failed: "failed",
  interrupted: "interrupted before finishing",
  incomplete: "incomplete outputs",
}[st.state] || st.state);

Results.badge = st => `<span class="status ${st.state === "complete" ? "complete" : st.state === "running" ? "running" : "other"}">
  ${st.state === "complete" ? "✓" : st.state === "running" ? "◔" : "○"} ${U.esc(Results.statusText(st))}</span>`;

Results.ORDER = () => {
  const cat = D.catalog;
  return [
    "original",
    ...cat.filter(c => c.group === "degradation").map(c => c.name),
    ...cat.filter(c => c.group === "enhancement").map(c => c.name),
    ...cat.filter(c => c.group === "degrade_enhance").map(c => c.name),
  ];
};

Results.render = () => {
  Results.key = Results.key || D.primary;
  U.$("#s-results").innerHTML = `
    <div class="wrap">
      <div class="section-head">
        <span class="eyebrow">Research results</span>
        <h2>What the experiments found</h2>
        <p>Each chart answers one question. The main score is <b>balanced accuracy</b>: the average hit rate over the seven
          diagnoses, so rare cancers count as much as common moles. Ranges show 95% confidence intervals from
          lesion-level bootstrap resampling.</p>
      </div>
      <div class="exp-switch" id="exp-switch">${D.order.map(k => {
        const e = U.exp(k);
        return `<button class="chip" type="button" aria-pressed="${k === Results.key}" data-k="${k}">${U.esc(e.label)}</button> ${Results.badge(e.status)}`;
      }).join("<span class='faint'>·</span>")}</div>
      <div id="results-body" class="charts-grid" style="margin-top:18px"></div>
    </div>`;
  U.$$("#exp-switch .chip").forEach(b => b.addEventListener("click", () => { Results.key = b.dataset.k; Results.render(); }));
  const e = U.exp(Results.key);
  if (e.status.state !== "complete") {
    U.$("#results-body").innerHTML = Results.statusCard(e) + Results.compareBlock();
    Results.drawCompare();
    return;
  }
  Results.body();
};

Results.statusCard = e => {
  const st = e.status;
  const prog = st.state === "running" && st.total ? `<div class="progress"><i style="width:${(100 * (st.done || 0) / st.total).toFixed(1)}%"></i></div>
    <span class="small num">${U.int(st.done)} of ${U.int(st.total)} predictions · last update ${U.esc(st.updated_utc || "")} UTC</span>` : "";
  return `<div class="status-card"><h3>${U.esc(e.label)}: ${U.esc(Results.statusText(st))}</h3>${prog}
    <p class="muted">No results are shown for this experiment until its run has finished and every output file has been checked
      (predictions for every image and condition, and all metric tables). ${st.detail ? `Detail: ${U.esc(st.detail)}` : ""}</p>
    <p class="small faint">Rebuild the website after the run finishes: <code>uv run skin-research build-site</code></p></div>`;
};

Results.q = (id, question, extra = "") => `<div class="qchart" id="${id}"><h3>${question}</h3>${extra}<div class="chart"></div><div class="answer"></div><div class="srcs"></div></div>`;

Results.body = () => {
  const k = Results.key;
  U.$("#results-body").innerHTML = [
    Results.q("q-overview", "How much does each change to the photo affect accuracy?"),
    Results.q("q-severity", "Does more severe damage reduce accuracy?", `<div class="controls" id="sev-metric"></div>`),
    `<div class="split">${Results.q("q-enh", "Does enhancing a good photo help?")}${Results.q("q-fix", "Can a fix recover performance after damage?")}</div>`,
    Results.q("q-class", "Which diagnosis is most affected by poor image quality?", `<div class="controls" id="class-filter"></div>`),
    Results.q("q-mel", "How well is melanoma detected as photos get worse?"),
    Results.q("q-calib", "Can the model's confidence be trusted?", `<div class="controls"><span class="label">Condition</span><select id="calib-select"></select></div>`),
    Results.q("q-cm", "Which diagnoses does the model confuse?", `<div class="controls"><span class="label">Condition</span><select id="cm-select"></select></div>`),
    Results.compareBlock(),
  ].join("");
  Results.overview(k); Results.severity(k); Results.enh(k); Results.fix(k);
  Results.classes(k); Results.melanoma(k); Results.calibration(k); Results.confusion(k); Results.drawCompare();
};

Results.set = (id, svg, answer, src) => {
  const box = U.$(`#${id}`);
  U.$(".chart", box).innerHTML = svg;
  U.$(".answer", box).innerHTML = answer;
  U.$(".srcs", box).innerHTML = src;
  return box;
};

/* ---- Q1 overview ---- */
Results.overview = k => {
  const r = U.results(k), o = U.orig(k);
  const names = Results.ORDER().filter(n => U.cond(k, n));
  const rowH = 17, L = 320, R = 70, T = 24, W = 960;
  const H = T + names.length * rowH + 28;
  const x = U.linear(0, 1, L, W - R);
  let s = "";
  [0, 0.25, 0.5, 0.75, 1].forEach(g => { s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T - 6}" y2="${H - 24}"/><text x="${x(g)}" y="${H - 8}" text-anchor="middle">${g}</text>`; });
  s += `<line class="base" x1="${x(o.bal)}" x2="${x(o.bal)}" y1="${T - 10}" y2="${H - 24}"/><text x="${x(o.bal) + 4}" y="${T - 12}" style="fill:var(--ink)">original ${U.f3(o.bal)}</text>`;
  names.forEach((n, i) => {
    const c = U.cond(k, n), yy = T + i * rowH + rowH / 2;
    const sig = n === "original" ? "" : c.d_hi < 0 ? " ▼" : c.d_lo > 0 ? " ▲" : "";
    s += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end" style="fill:var(--ink)">${U.esc(U.title(n))}${sig}</text>`;
    s += `<line x1="${x(c.bal_lo)}" x2="${x(c.bal_hi)}" y1="${yy}" y2="${yy}" stroke="var(--ink-3)" stroke-width="2"/>`;
    s += `<circle cx="${x(c.bal)}" cy="${yy}" r="5" fill="${U.groupColor(n)}" stroke="#fff" stroke-width="1.5"/>`;
    s += `<text x="${W - R + 8}" y="${yy + 4}" class="num">${U.f3(c.bal)}</text>`;
    s += `<rect x="0" y="${yy - rowH / 2}" width="${W}" height="${rowH}" fill="transparent" data-n="${n}"/>`;
  });
  const others = r.conditions.filter(c => c.name !== "original");
  const worse = others.filter(c => c.d_hi < 0), better = others.filter(c => c.d_lo > 0);
  const worst = [...others].sort((a, b) => a.bal - b.bal)[0];
  const box = Results.set("q-overview",
    U.svg(W, H, "Balanced accuracy with 95% confidence interval for every condition", s) +
    `<div class="legend">${Object.entries(U.GROUPS).map(([, g]) => `<span><i class="dot" style="background:${g.color}"></i>${g.name}</span>`).join("")}
      <span>▼ / ▲ = measurably lower / higher than original (95% CI of the change excludes 0)</span></div>`,
    `<b>${worse.length} of ${others.length}</b> changes made the AI measurably less accurate and <b>${better.length}</b> made it
     measurably more accurate. The largest drop was <b>${U.esc(U.title(worst.name))}</b>: balanced accuracy ${U.f3(worst.bal)}
     versus ${U.f3(o.bal)} on the original photos.`,
    U.source(k, "summary", "balanced_accuracy, _ci_low, _ci_high") + U.source(k, "vs_original", "delta_balanced_accuracy_ci_*"));
  U.$$("rect[data-n]", box).forEach(rc => {
    const c = U.cond(k, rc.dataset.n);
    U.hover(rc, `<b>${U.esc(U.title(c.name))}</b><br>balanced accuracy ${U.f3(c.bal)} (95% CI ${U.f3(c.bal_lo)}–${U.f3(c.bal_hi)})<br>accuracy ${U.f3(c.acc)}`
      + (c.name !== "original" ? `<br>change ${U.signed(c.d_bal)} (CI ${U.signed(c.d_lo, 2)} to ${U.signed(c.d_hi, 2)})<br>answer changed for ${U.pct(c.flip)} of photos` : ""));
  });
};

/* ---- Q2 severity ---- */
Results.SEVM = {
  bal: ["Balanced accuracy", c => c.bal, c => [c.bal_lo, c.bal_hi]],
  acc: ["Accuracy", c => c.acc, c => [c.acc_lo, c.acc_hi]],
  mel: ["Melanoma sensitivity", (c, k) => ((U.results(k).per_class[c.name] || {}).mel || {}).recall, null],
};
Results.sevMetric = "bal";
Results.severity = k => {
  const ctl = U.$("#sev-metric");
  ctl.innerHTML = Object.entries(Results.SEVM).map(([m, [n]]) => `<button class="chip" type="button" aria-pressed="${m === Results.sevMetric}" data-m="${m}">${n}</button>`).join("");
  U.$$("button", ctl).forEach(b => b.addEventListener("click", () => { Results.sevMetric = b.dataset.m; Results.severity(k); }));
  const [label, get, ci] = Results.SEVM[Results.sevMetric];
  const types = [...new Set(D.catalog.filter(c => c.group === "degradation").map(c => c.degradation))];
  const sev = ["mild", "moderate", "severe"];
  const W = 230, H = 160, L = 30, R = 34, T = 10, B = 24;
  const x = i => L + i * (W - L - R) / 3, y = U.linear(0, 1, H - B, T);
  let mono = 0;
  const panels = types.map(d => {
    const pts = ["original", ...sev.map(s => `${d}__${s}`)].map(n => U.cond(k, n)).filter(Boolean);
    const vals = pts.map(p => get(p, k));
    if (vals.every((v, i) => i === 0 || v <= vals[i - 1] + 1e-9)) mono += 1;
    let s = "";
    [0, 0.5, 1].forEach(g => { s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}"/><text x="${L - 5}" y="${y(g) + 4}" text-anchor="end">${g}</text>`; });
    if (ci) {
      const up = pts.map((p, i) => `${x(i)},${y(ci(p)[1])}`).join(" ");
      const dn = pts.map((p, i) => `${x(i)},${y(ci(p)[0])}`).reverse().join(" ");
      s += `<polygon points="${up} ${dn}" fill="var(--accent-soft)"/>`;
    }
    s += `<polyline points="${vals.map((v, i) => `${x(i)},${y(v)}`).join(" ")}" fill="none" stroke="var(--c-deg)" stroke-width="2"/>`;
    vals.forEach((v, i) => {
      s += `<circle cx="${x(i)}" cy="${y(v)}" r="4.5" fill="${i ? "var(--c-deg)" : "var(--c-orig)"}" stroke="#fff" stroke-width="1.5"/>`;
      s += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${i ? sev[i - 1].slice(0, 3) : "orig"}</text>`;
      s += `<rect x="${x(i) - 14}" y="${T}" width="28" height="${H - B - T}" fill="transparent" data-i="${i}"/>`;
    });
    s += `<text x="${x(vals.length - 1) + 7}" y="${y(vals[vals.length - 1]) + 4}" style="fill:var(--ink);font-weight:700">${U.f2(vals[vals.length - 1])}</text>`;
    return { d, pts, vals, svg: U.svg(W, H, `${U.NAMES[d]}: ${label} by severity`, s) };
  });
  const box = Results.set("q-severity",
    `<div class="multiples">${panels.map(p => `<div class="mini" data-d="${p.d}"><h4>${U.NAMES[p.d]}</h4>${p.svg}</div>`).join("")}</div>`,
    `${label} fell steadily from original to severe for <b>${mono} of ${types.length}</b> damage types.
     ${Results.sevMetric === "bal" ? "Shaded bands are 95% confidence intervals: where a band still overlaps the original value, that strength of damage has not been shown to matter." : ""}`,
    U.source(k, Results.sevMetric === "mel" ? "per_class" : "summary", Results.sevMetric === "mel" ? "recall (mel)" : "accuracy / balanced_accuracy with CIs"));
  panels.forEach(p => U.$$(`.mini[data-d="${p.d}"] rect`, box).forEach(rc => {
    const c = p.pts[+rc.dataset.i];
    U.hover(rc, `<b>${U.esc(U.title(c.name))}</b><br>${label}: ${U.f3(p.vals[+rc.dataset.i])}${ci ? ` (CI ${U.f3(ci(c)[0])}–${U.f3(ci(c)[1])})` : ""}<br><span>${U.esc(U.params(c.name))}</span>`);
  }));
};

/* ---- Q3 enhancement of good photos ---- */
Results.enh = k => {
  const o = U.orig(k);
  const rows = [o, ...D.catalog.filter(c => c.group === "enhancement").map(c => U.cond(k, c.name)).filter(Boolean)];
  const W = 440, rowH = 30, L = 150, R = 50, T = 8, H = T + rows.length * rowH + 26;
  const x = U.linear(0, 1, L, W - R);
  let s = "";
  [0, 0.5, 1].forEach(g => { s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - 22}"/><text x="${x(g)}" y="${H - 6}" text-anchor="middle">${g}</text>`; });
  rows.forEach((c, i) => {
    const yy = T + i * rowH;
    s += `<text x="${L - 8}" y="${yy + 19}" text-anchor="end" style="fill:var(--ink)">${U.esc(c.name === "original" ? "Original" : U.title(c.name))}</text>`;
    s += `<rect x="${L}" y="${yy + 7}" width="${x(c.bal) - L}" height="15" rx="3" fill="${U.groupColor(c.name)}"/>`;
    s += `<line x1="${x(c.bal_lo)}" x2="${x(c.bal_hi)}" y1="${yy + 14.5}" y2="${yy + 14.5}" stroke="var(--ink)" stroke-width="1.5"/>`;
    s += `<text x="${x(Math.max(c.bal, c.bal_hi)) + 5}" y="${yy + 19}">${U.f2(c.bal)}</text>`;
    s += `<rect x="0" y="${yy}" width="${W}" height="${rowH}" fill="transparent" data-n="${c.name}"/>`;
  });
  s += `<line class="base" x1="${x(o.bal)}" x2="${x(o.bal)}" y1="${T}" y2="${H - 22}"/>`;
  const enh = rows.slice(1);
  const worse = enh.filter(c => c.d_hi < 0), better = enh.filter(c => c.d_lo > 0);
  const clahe = U.cond(k, "enhance__clahe");
  const box = Results.set("q-enh", U.svg(W, H, "Balanced accuracy after enhancing original photos", s),
    `${better.length ? `<b>${better.length}</b> enhancement(s) measurably helped.` : "<b>None</b> of the enhancements measurably improved on the original photos."}
     <b>${worse.length}</b> made the AI measurably worse${worse.length ? ` (${worse.map(c => U.esc(U.title(c.name))).join(", ")})` : ""}.
     ${clahe ? `CLAHE changed balanced accuracy by ${U.signed(clahe.d_bal)} (CI ${U.signed(clahe.d_lo, 2)} to ${U.signed(clahe.d_hi, 2)}).` : ""}
     The model was trained on unenhanced photos, so "nicer-looking" images can look less familiar to it.`,
    U.source(k, "summary", "balanced_accuracy") + U.source(k, "vs_original", "delta CIs"));
  U.$$("rect[data-n]", box).forEach(rc => {
    const c = U.cond(k, rc.dataset.n);
    U.hover(rc, `<b>${U.esc(U.title(c.name))}</b><br>${U.f3(c.bal)} (CI ${U.f3(c.bal_lo)}–${U.f3(c.bal_hi)})${c.name !== "original" ? `<br>change ${U.signed(c.d_bal)}; answers changed ${U.pct(c.flip)}` : ""}`);
  });
};

/* ---- Q4 restoration ---- */
Results.fix = k => {
  const rr = U.results(k).restoration, o = U.orig(k);
  const W = 480, rowH = 28, L = 230, R = 16, T = 8, H = T + rr.length * rowH + 26;
  const x = U.linear(0, 1, L, W - R);
  let s = "";
  [0, 0.5, 1].forEach(g => { s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - 22}"/><text x="${x(g)}" y="${H - 6}" text-anchor="middle">${g}</text>`; });
  s += `<line class="base" x1="${x(o.bal)}" x2="${x(o.bal)}" y1="${T}" y2="${H - 22}"/>`;
  rr.forEach((f, i) => {
    const yy = T + i * rowH + 14, c = U.catalog[f.name];
    const col = f.lo > 0 ? "var(--ok)" : f.hi < 0 ? "var(--bad)" : "var(--ink-3)";
    s += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end" style="fill:var(--ink)">${U.esc(U.NAMES[c.degradation])} → ${U.esc(U.NAMES[c.enhancement])}</text>`;
    s += `<line x1="${x(f.deg)}" x2="${x(f.enh)}" y1="${yy}" y2="${yy}" stroke="${col}" stroke-width="2.5"/>`;
    s += `<circle cx="${x(f.deg)}" cy="${yy}" r="5" fill="var(--c-deg)" stroke="#fff" stroke-width="1.5"/>`;
    s += `<circle cx="${x(f.enh)}" cy="${yy}" r="5" fill="var(--c-fix)" stroke="#fff" stroke-width="1.5"/>`;
    s += `<rect x="0" y="${yy - 14}" width="${W}" height="${rowH}" fill="transparent" data-i="${i}"/>`;
  });
  const helped = rr.filter(f => f.lo > 0), hurt = rr.filter(f => f.hi < 0);
  const best = [...rr].sort((a, b) => (b.recovery ?? -9) - (a.recovery ?? -9))[0];
  const box = Results.set("q-fix", U.svg(W, H, "Balanced accuracy before and after fixing damaged photos", s) +
    `<div class="legend"><span><i class="dot" style="background:var(--c-deg)"></i>damaged</span><span><i class="dot" style="background:var(--c-fix)"></i>damaged, then fixed</span>
     <span>line: green = measurably better, red = measurably worse, grey = unclear</span></div>`,
    `<b>${helped.length} of ${rr.length}</b> fixes measurably helped and <b>${hurt.length}</b> made things worse. The best was
     <b>${U.esc(U.title(best.name))}</b>, recovering ${U.pct(best.recovery)} of the accuracy lost to the damage. Fixes work when
     they undo the specific damage; mismatched fixes add new problems.`,
    U.source(k, "restoration", "balacc_degraded, balacc_enhanced, recovery_fraction"));
  U.$$("rect[data-i]", box).forEach(rc => {
    const f = rr[+rc.dataset.i];
    U.hover(rc, `<b>${U.esc(U.title(f.name))}</b><br>damaged ${U.f3(f.deg)} → fixed ${U.f3(f.enh)} (original ${U.f3(f.orig)})<br>change CI ${U.signed(f.lo, 2)} to ${U.signed(f.hi, 2)}<br>recovered ${U.pct(f.recovery)} of the loss`);
  });
};

/* ---- Q5 class heatmap ---- */
Results.classGroup = "degradation";
Results.classes = k => {
  const r = U.results(k), counts = r.meta.class_counts;
  const ctl = U.$("#class-filter");
  ctl.innerHTML = ["degradation", "enhancement", "degrade_enhance"].map(g =>
    `<button class="chip" type="button" aria-pressed="${g === Results.classGroup}" data-g="${g}"><i class="dot" style="background:${U.GROUPS[g].color}"></i>${U.GROUPS[g].name}</button>`).join("");
  U.$$("button", ctl).forEach(b => b.addEventListener("click", () => { Results.classGroup = b.dataset.g; Results.classes(k); }));
  const names = ["original", ...D.catalog.filter(c => c.group === Results.classGroup).map(c => c.name)];
  const pc = r.per_class;
  const table = `<div class="scroll"><table class="t heat"><thead><tr><th>Condition</th>${D.labels.map(l =>
    `<th class="c" title="${U.esc(U.label(l))}">${l}<div class="small faint">n=${counts[l] ?? 0}</div></th>`).join("")}</tr></thead><tbody>${names.map(n =>
    `<tr><td>${U.esc(U.title(n))}</td>${D.labels.map(l => {
      const v = (pc[n] && pc[n][l]) ? pc[n][l].recall : null;
      return `<td class="v" ${U.seqCell(v)} data-n="${n}" data-l="${l}">${U.f2(v)}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>`;
  // which diagnosis loses most sensitivity on average across all damage conditions (ignoring classes with < 5 photos)
  const dmg = D.catalog.filter(c => c.group === "degradation").map(c => c.name);
  const drops = D.labels.filter(l => (counts[l] || 0) >= 5).map(l => ({
    l, drop: pc.original[l].recall - dmg.reduce((a, n) => a + (pc[n][l].recall ?? 0), 0) / dmg.length }))
    .sort((a, b) => b.drop - a.drop);
  const rare = D.labels.filter(l => (counts[l] || 0) < 5);
  const box = Results.set("q-class", table,
    `Averaged over all ${dmg.length} damage conditions, <b>${U.esc(U.label(drops[0].l))}</b> lost the most sensitivity
     (${U.f2(pc.original[drops[0].l].recall)} on originals, ${U.signed(-drops[0].drop, 2)} on average), followed by
     ${U.esc(U.label(drops[1].l))}. ${rare.length ? `Columns ${rare.join(" and ")} have fewer than 5 photos, so their values jump between 0 and 1 and are not compared.` : ""}`,
    U.source(k, "per_class", "recall (sensitivity) per condition × diagnosis"));
  U.$$("td[data-n]", box).forEach(td => {
    const v = pc[td.dataset.n][td.dataset.l];
    U.hover(td, `<b>${U.esc(U.label(td.dataset.l))}</b> · ${U.esc(U.title(td.dataset.n))}<br>sensitivity ${U.f2(v.recall)} · specificity ${U.f2(v.specificity)}<br>${v.support} photo(s) of this diagnosis`);
  });
};

/* ---- Q6 melanoma ---- */
Results.melanoma = k => {
  const r = U.results(k), pc = r.per_class, n = r.meta.class_counts.mel || 0;
  const types = [...new Set(D.catalog.filter(c => c.group === "degradation").map(c => c.degradation))];
  const sev = ["mild", "moderate", "severe"], cols = ["original", ...sev];
  const colColor = ["var(--c-orig)", "#86b6ef", "#3987e5", "#184f95"];
  const W = 640, rowH = 26, L = 140, R = 20, T = 22, H = T + types.length * rowH + 24;
  const x = U.linear(0, 1, L, W - R);
  const o = pc.original.mel.recall;
  let s = `<line class="base" x1="${x(o)}" x2="${x(o)}" y1="${T - 6}" y2="${H - 22}"/><text x="${x(o) + 4}" y="${T - 10}" style="fill:var(--ink)">original ${U.f2(o)}</text>`;
  [0, 0.25, 0.5, 0.75, 1].forEach(g => { s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - 22}"/><text x="${x(g)}" y="${H - 6}" text-anchor="middle">${U.pct(g)}</text>`; });
  types.forEach((d, i) => {
    const yy = T + i * rowH + 12;
    const vals = sev.map(sv => pc[`${d}__${sv}`].mel.recall);
    s += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end" style="fill:var(--ink)">${U.NAMES[d]}</text>`;
    s += `<line x1="${x(Math.min(o, ...vals))}" x2="${x(Math.max(o, ...vals))}" y1="${yy}" y2="${yy}" stroke="var(--rule)" stroke-width="6" stroke-linecap="round"/>`;
    vals.forEach((v, j) => { s += `<circle cx="${x(v)}" cy="${yy}" r="6" fill="${colColor[j + 1]}" stroke="#fff" stroke-width="1.5"/>`; });
    s += `<rect x="0" y="${yy - 13}" width="${W}" height="${rowH}" fill="transparent" data-d="${d}"/>`;
  });
  const mild = types.map(d => ({ d, v: pc[`${d}__mild`].mel.recall })).sort((a, b) => a.v - b.v);
  const zero = types.filter(d => pc[`${d}__severe`].mel.recall === 0);
  const box = Results.set("q-mel", U.svg(W, H, "Melanoma sensitivity by damage type and strength", s) +
    `<div class="legend">${cols.slice(1).map((c, j) => `<span><i class="dot" style="background:${colColor[j + 1]}"></i>${c}</span>`).join("")}</div>`,
    `The AI found <b>${U.pct(o)}</b> of the ${n} melanomas in original photos. Even <b>mild</b> ${U.esc(U.NAMES[mild[0].d].toLowerCase())}
     cut this to <b>${U.pct(mild[0].v)}</b>. ${zero.length ? `At severe strength, ${zero.length} damage type(s) left <b>no melanoma detected</b> (${zero.map(d => U.NAMES[d]).join(", ")}).` : ""}
     Missed melanomas are the most costly mistake a skin-cancer tool can make. With ${n} melanomas, each one is ${U.pct(1 / n, 1)} of this score.`,
    U.source(k, "per_class", "recall for label mel"));
  U.$$("rect[data-d]", box).forEach(rc => {
    const d = rc.dataset.d;
    U.hover(rc, `<b>${U.NAMES[d]}</b> · melanoma sensitivity<br>${cols.map(c => `${c}: ${U.pct(c === "original" ? o : pc[`${d}__${c}`].mel.recall)}`).join("<br>")}`);
  });
};

/* ---- Q7 calibration ---- */
Results.calibration = k => {
  const r = U.results(k);
  const sel = U.$("#calib-select");
  sel.innerHTML = Results.ORDER().map(n => `<option value="${n}">${U.esc(U.title(n))}</option>`).join("");
  const draw = n => {
    const c = U.cond(k, n), bins = r.reliability[n] || [], hist = r.confidence_hist[n];
    const W = 340, H = 240, L = 40, R = 10, T = 10, B = 36;
    const x = U.linear(0, 1, L, W - R), y = U.linear(0, 1, H - B, T);
    let s = "";
    [0, 0.5, 1].forEach(g => { s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - B}"/><line class="grid" x1="${L}" x2="${W - R}" y1="${y(g)}" y2="${y(g)}"/><text x="${x(g)}" y="${H - B + 14}" text-anchor="middle">${U.pct(g)}</text><text x="${L - 5}" y="${y(g) + 4}" text-anchor="end">${U.pct(g)}</text>`; });
    s += `<line class="base" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>`;
    s += `<polyline points="${bins.map(b => `${x(b.confidence)},${y(b.accuracy)}`).join(" ")}" fill="none" stroke="var(--accent)" stroke-width="2"/>`;
    bins.forEach(b => { s += `<circle cx="${x(b.confidence)}" cy="${y(b.accuracy)}" r="${Math.max(3, Math.min(9, Math.sqrt(b.count)))}" fill="var(--accent)" stroke="#fff" stroke-width="1.5"><title>${b.count} answers: confidence ${U.pct(b.confidence)}, right ${U.pct(b.accuracy)}</title></circle>`; });
    s += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">confidence</text>`;
    s += `<text x="11" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 11 ${(T + H - B) / 2})">actually right</text>`;
    // histogram: right vs wrong by confidence bin
    const W2 = 340, H2 = 240, L2 = 34, B2 = 36, T2 = 10;
    const max = Math.max(1, ...hist.correct, ...hist.wrong);
    const bw = (W2 - L2 - 10) / hist.correct.length;
    const y2 = U.linear(0, max, H2 - B2, T2);
    let h = `<line class="grid" x1="${L2}" x2="${W2 - 10}" y1="${H2 - B2}" y2="${H2 - B2}"/>`;
    [0, Math.round(max / 2), max].forEach(g => { h += `<text x="${L2 - 5}" y="${y2(g) + 4}" text-anchor="end">${g}</text>`; });
    hist.correct.forEach((v, i) => {
      const x0 = L2 + i * bw;
      h += `<rect x="${x0 + 1}" y="${y2(v)}" width="${bw / 2 - 1.5}" height="${H2 - B2 - y2(v)}" fill="var(--ok)"><title>${v} right answers at ${i * 10}–${i * 10 + 10}% confidence</title></rect>`;
      h += `<rect x="${x0 + bw / 2}" y="${y2(hist.wrong[i])}" width="${bw / 2 - 1.5}" height="${H2 - B2 - y2(hist.wrong[i])}" fill="var(--bad)"><title>${hist.wrong[i]} wrong answers at ${i * 10}–${i * 10 + 10}% confidence</title></rect>`;
      if (i % 2 === 0) h += `<text x="${x0}" y="${H2 - B2 + 14}" text-anchor="middle">${i * 10}%</text>`;
    });
    h += `<text x="${(L2 + W2) / 2}" y="${H2 - 4}" text-anchor="middle">confidence</text>`;
    Results.set("q-calib",
      `<div class="split"><div>${U.svg(W, H, `Reliability diagram, ${U.title(n)}`, s)}<p class="small faint">Dots on the dashed line would mean confidence matches reality.</p></div>
       <div>${U.svg(W2, H2, `Confidence of right and wrong answers, ${U.title(n)}`, h)}<div class="legend"><span><i class="dot" style="background:var(--ok)"></i>right</span><span><i class="dot" style="background:var(--bad)"></i>wrong</span></div></div></div>`,
      `${U.esc(U.title(n))}: average confidence <b>${U.pct(c.conf)}</b> while accuracy was <b>${U.pct(c.acc)}</b>, so the model is
       <b>${c.conf < c.acc ? "under-confident" : "over-confident"}</b> here (calibration error ${U.f3(c.ece)}). Right answers averaged
       ${U.pct(c.conf_ok)} confidence and wrong answers ${U.pct(c.conf_bad)}: the two overlap heavily, so a single confidence value
       cannot tell you whether a particular answer is right.`,
      U.source(k, "details", "reliability_bins") + U.source(k, "predictions", "confidence, correct (histogram)") + U.source(k, "summary", "ece, mean_confidence"));
  };
  sel.addEventListener("change", () => draw(sel.value));
  draw("original");
};

/* ---- Q8 confusion matrix ---- */
Results.confusion = k => {
  const r = U.results(k);
  const sel = U.$("#cm-select");
  sel.innerHTML = Results.ORDER().map(n => `<option value="${n}">${U.esc(U.title(n))}</option>`).join("");
  const draw = n => {
    const cm = r.confusion[n];
    let top = { v: 0 };
    const rows = cm.map((row, i) => {
      const tot = row.reduce((a, b) => a + b, 0);
      return `<tr><th>${D.labels[i]} <span class="faint small">(${tot})</span></th>${row.map((v, j) => {
        const p = tot ? v / tot : null;
        if (i !== j && v > top.v) top = { v, i, j };
        return `<td class="v" ${U.seqCell(p)} title="${v} ${D.labels[i]} photo(s) predicted as ${D.labels[j]}">${v}</td>`;
      }).join("")}</tr>`;
    }).join("");
    Results.set("q-cm", `<div class="scroll"><table class="t cm"><thead><tr><th>true ↓ / predicted →</th>${D.labels.map(l => `<th class="c">${l}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
      <p class="small faint">Numbers are photo counts. Shading is the share of each true diagnosis (row) given each answer; the diagonal is correct.</p>`,
      top.v ? `${U.esc(U.title(n))}: the most common mistake was calling <b>${top.v}</b> ${U.esc(U.label(D.labels[top.i]))} photo(s)
        <b>${U.esc(U.label(D.labels[top.j]))}</b>.` : "No off-diagonal mistakes.",
      U.source(k, "details", `per_condition.${n}.confusion_matrix`));
  };
  sel.addEventListener("change", () => draw(sel.value));
  draw("original");
};

/* ---- Q9 validation vs test ---- */
Results.compareBlock = () => `<div class="qchart" id="q-compare"><h3>Do the findings hold on a second, larger set of photos?</h3><div class="chart"></div><div class="answer"></div><div class="srcs"></div></div>`;
Results.drawCompare = () => {
  const keys = D.order;
  if (keys.length < 2) return;
  const [a, b] = keys;
  const ea = U.exp(a), eb = U.exp(b);
  if (!U.isComplete(a) || !U.isComplete(b)) {
    const pending = [ea, eb].filter(e => e.status.state !== "complete");
    Results.set("q-compare", pending.map(e => `<p>${U.esc(e.label)}: ${Results.badge(e.status)}</p>`).join(""),
      `The comparison appears once both experiments are complete. Until then, no results from ${pending.map(e => U.esc(e.label)).join(" or ")} are shown or mixed with the others.`, "");
    return;
  }
  const names = Results.ORDER().filter(n => U.cond(a, n) && U.cond(b, n));
  const rowH = 16, L = 320, R = 20, T = 18, W = 960, H = T + names.length * rowH + 24;
  const x = U.linear(0, 1, L, W - R);
  let s = "";
  [0, 0.25, 0.5, 0.75, 1].forEach(g => { s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="${T}" y2="${H - 22}"/><text x="${x(g)}" y="${H - 6}" text-anchor="middle">${g}</text>`; });
  let same = 0;
  names.forEach((n, i) => {
    const ca = U.cond(a, n), cb = U.cond(b, n), yy = T + i * rowH + 8;
    if (n !== "original" && Math.sign(ca.d_bal) === Math.sign(cb.d_bal)) same += 1;
    s += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end" style="fill:var(--ink)">${U.esc(U.title(n))}</text>`;
    s += `<line x1="${x(ca.bal)}" x2="${x(cb.bal)}" y1="${yy}" y2="${yy}" stroke="var(--rule)" stroke-width="3"/>`;
    s += `<circle cx="${x(ca.bal)}" cy="${yy}" r="4.5" fill="var(--c-enh)"><title>${U.esc(ea.label)}: ${U.f3(ca.bal)}</title></circle>`;
    s += `<rect x="${x(cb.bal) - 4.5}" y="${yy - 4.5}" width="9" height="9" fill="var(--c-deg)"><title>${U.esc(eb.label)}: ${U.f3(cb.bal)}</title></rect>`;
  });
  Results.set("q-compare", U.svg(W, H, "Balanced accuracy on both datasets", s) +
    `<div class="legend"><span><i class="dot" style="background:var(--c-enh)"></i>${U.esc(ea.label)} (${U.int(ea.results.meta.n_images)} photos)</span><span><i class="dot" style="background:var(--c-deg);border-radius:2px"></i>${U.esc(eb.label)} (${U.int(eb.results.meta.n_images)} photos)</span></div>`,
    `The direction of the change versus the original photo agreed on both datasets for <b>${same} of ${names.length - 1}</b> conditions.
     Each dataset is evaluated separately; their photos are never pooled.`,
    U.source(a, "summary", "balanced_accuracy") + U.source(b, "summary", "balanced_accuracy"));
};

window.Results = Results;
