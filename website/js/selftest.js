/* Browser self-test, loaded only when the URL contains ?selftest.
   Clicks through every section and control, checks the rendered values against SITE_DATA, loads every
   image path and records JavaScript errors. The JSON result is written to <pre id="selftest-result">,
   which `scripts/check_website.py` reads from a headless browser. */
"use strict";

(async function selftest() {
  const R = { errors: [], checks: 0, failures: [], images: { checked: 0, failed: [] }, steps: [] };
  const fail = msg => R.failures.push(msg);
  const ok = (cond, msg) => { R.checks += 1; if (!cond) fail(msg); };
  window.addEventListener("error", e => R.errors.push(`${e.message} @ ${e.filename}:${e.lineno}`));
  const origErr = console.error;
  console.error = (...a) => { R.errors.push(a.map(String).join(" ")); origErr(...a); };
  const tick = (ms = 0) => new Promise(r => setTimeout(r, ms));
  const click = el => { el.dispatchEvent(new MouseEvent("click", { bubbles: true })); };

  try {
    /* 1. navigation */
    for (const a of U.$$(".tabs a")) {
      const id = a.getAttribute("href").slice(1);
      location.hash = id; await tick(30);
      const sec = U.$(`#s-${id}`);
      ok(!sec.hidden, `section ${id} not visible after navigation`);
      ok(sec.textContent.trim().length > 200, `section ${id} looks empty`);
      ok(!sec.textContent.includes("could not be displayed"), `section ${id} failed to render`);
      ok(U.$$("main > section").filter(s => !s.hidden).length === 1, `more than one section visible on ${id}`);
    }
    R.steps.push("navigation");

    /* 2. image lab: every lesion × every transformation × every severity */
    location.hash = "lab"; await tick(20);
    let combos = 0;
    for (const lb of U.$$("#lesions .lesion-btn")) {
      click(lb); await tick();
      for (const chip of U.$$("#tgroups .chip")) {
        click(chip); await tick();
        const sev = U.$$("#severity button").filter(b => !b.disabled);
        for (const sb of (sev.length ? sev : [null])) {
          if (sb) { click(sb); await tick(); }
          combos += 1;
          const cond = Lab.condName(), les = U.lesion(lb.dataset.id), rec = les.conditions[cond];
          ok(!!rec, `lab: no data for ${les.id}/${cond}`);
          const imgs = U.$$("#lab-images img").map(i => i.getAttribute("src"));
          ok(imgs.includes(rec.img) && imgs.includes(les.conditions.original.img), `lab: wrong images for ${les.id}/${cond}`);
          const txt = U.$("#lab-preds").textContent;
          ok(txt.includes(rec.pred) && txt.includes(U.pct(rec.conf, 1)), `lab: prediction text mismatch ${les.id}/${cond}`);
          ok(U.$("#lab-title").textContent === U.title(cond), `lab: title mismatch ${cond}`);
        }
      }
    }
    R.lab_combinations = combos;
    for (const v of U.$$("#viewmode button")) {
      click(v); await tick();
      ok(v.dataset.v !== "slider" || !!U.$("#slider input[type=range]"), "lab: slider view missing");
    }
    const range = U.$("#slider input");
    if (range) { range.value = 20; range.dispatchEvent(new Event("input")); ok(U.$("#slider .top").style.clipPath.includes("20%"), "lab: slider does not move"); }
    for (const c of U.$$("#gallery .case")) {
      click(c); await tick();
      const s = D.samples.stories[+c.dataset.i];
      ok(Lab.state.lesion === s.image_id, `gallery: case ${s.key} did not load its lesion`);
      if (s.condition !== "original") ok(Lab.condName() === s.condition, `gallery: case ${s.key} did not load ${s.condition}`);
    }
    R.steps.push("lab");

    /* 3. prediction explorer */
    location.hash = "explorer"; await tick(20);
    for (const lb of U.$$("#ex-lesions .chip")) {
      click(lb); await tick();
      for (const f of U.$$("#ex-filter .chip")) {
        click(f); await tick();
        const want = D.catalog.filter(c => f.dataset.g === "all" || c.name === "original" || c.group === f.dataset.g).length;
        ok(U.$$("#ex-grid .cond-card").length === want, `explorer: filter ${f.dataset.g} shows wrong count`);
      }
      for (const card of U.$$("#ex-grid .cond-card")) {
        click(card); await tick();
        const rec = U.lesion(lb.dataset.id).conditions[card.dataset.c];
        ok(U.$("#ex-main").textContent.includes(rec.pred) && U.$("#ex-main").textContent.includes(U.pct(rec.conf, 1)),
          `explorer: wrong prediction shown for ${lb.dataset.id}/${card.dataset.c}`);
      }
    }
    ok(!!U.$("#ex-reliability svg") && !!U.$("#ex-confgap svg"), "explorer: confidence charts missing");
    R.steps.push("explorer");

    /* 4. results for every experiment */
    location.hash = "results"; await tick(20);
    for (const key of D.order) {
      const btn = U.$$("#exp-switch .chip").find(b => b.dataset.k === key);
      click(btn); await tick(10);
      const complete = U.isComplete(key);
      if (!complete) {
        ok(!!U.$("#results-body .status-card"), `results: ${key} incomplete but no status card`);
        ok(!U.$("#q-overview"), `results: ${key} incomplete but charts shown`);
        continue;
      }
      for (const id of ["q-overview", "q-severity", "q-enh", "q-fix", "q-class", "q-mel", "q-calib", "q-cm", "q-compare"]) {
        const box = U.$(`#${id}`);
        ok(!!box && !!U.$(".chart", box).innerHTML.trim() && !!U.$(".answer", box).textContent.trim(), `results: ${key}/${id} empty`);
        if (id !== "q-compare") ok(box.textContent.includes(U.results(key).meta.experiment), `results: ${key}/${id} has no source for this experiment`);
      }
      const o = U.orig(key);
      ok(U.$("#q-overview").textContent.includes(U.f3(o.bal)), `results: ${key} overview does not show original balanced accuracy`);
      for (const b of U.$$("#sev-metric button")) { click(b); await tick(); ok(U.$$("#q-severity .mini").length === 8, "results: severity panels missing"); }
      for (const b of U.$$("#class-filter button")) { click(b); await tick(); ok(U.$$("#q-class tbody tr").length > 1, "results: heatmap empty"); }
      for (const sel of ["#calib-select", "#cm-select"]) {
        const s = U.$(sel);
        for (const opt of [...s.options]) { s.value = opt.value; s.dispatchEvent(new Event("change")); await tick(); }
        ok(s.options.length === D.catalog.length, `results: ${sel} lists ${s.options.length} conditions`);
      }
    }
    R.steps.push("results");

    /* 5. about */
    location.hash = "about"; await tick(20);
    ok(U.$$(".about-body section").length >= 10, "about: sections missing");
    for (const a of U.$$(".toc a")) ok(!!document.getElementById(a.getAttribute("href").slice(1)), `about: broken link ${a.getAttribute("href")}`);
    R.steps.push("about");

    /* 6. guided tour */
    click(U.$("[data-tour]")); await tick();
    ok(!U.$("#tour").hidden, "tour: did not open");
    const n = Tour.slides().length;
    for (let i = 1; i < n; i++) { click(U.$("#tour-next")); await tick(); }
    ok(U.$("#tour-count").textContent.startsWith(`${n} / ${n}`), "tour: did not reach last slide");
    click(U.$("#tour-prev")); await tick();
    ok(U.$("#tour-count").textContent.startsWith(`${n - 1} / ${n}`), "tour: previous does not work");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" })); await tick();
    ok(U.$("#tour").hidden, "tour: Escape does not close");
    R.tour_slides = n;
    R.steps.push("tour");

    /* 7. layout: nothing may stick out past the right edge of the screen (checked in every section) */
    R.viewport = window.innerWidth;
    R.overflow = [];
    for (const id of ["overview", "how", "lab", "explorer", "results", "about"]) {
      location.hash = id; await tick(30);
      const vw = document.documentElement.clientWidth;
      for (const el of U.$$(`#s-${id} *, .topbar *`)) {
        if (el.closest(".scroll, svg, .tabs, .toc")) continue;     // horizontal scroll strips may be wider by design
        const r = el.getBoundingClientRect();
        if (r.width && r.right > vw + 1) {
          R.overflow.push(`${id}: <${el.tagName.toLowerCase()} class="${el.className}"> right=${Math.round(r.right)} > ${vw}`);
          if (R.overflow.length > 40) break;
        }
      }
      ok(document.documentElement.scrollWidth <= vw + 1, `layout: ${id} scrolls sideways at ${vw}px (page ${document.documentElement.scrollWidth}px)`);
    }
    R.steps.push("layout");

    /* 8. every exported image path resolves */
    const paths = new Set();
    D.samples.lesions.forEach(l => { paths.add(l.thumb); Object.values(l.conditions).forEach(c => c.img && paths.add(c.img)); });
    const results = await Promise.all([...paths].map(src => new Promise(res => {
      const im = new Image();
      im.onload = () => res(im.naturalWidth > 0 ? null : src);
      im.onerror = () => res(src);
      im.src = src;
    })));
    R.images.checked = paths.size;
    R.images.failed = results.filter(Boolean);
    R.steps.push("images");
  } catch (err) {
    R.errors.push(`selftest crashed: ${err.message}\n${err.stack}`);
  }
  R.passed = R.failures.length === 0 && R.errors.length === 0 && R.images.failed.length === 0;
  const pre = document.createElement("pre");
  pre.id = "selftest-result";
  pre.textContent = JSON.stringify(R);
  document.body.appendChild(pre);
})();
