/* Boot + navigation: one section visible at a time, chosen by the URL hash (#overview … #about). */
"use strict";

(function () {
  const SECTIONS = ["overview", "how", "lab", "explorer", "results", "about"];
  const RENDER = { overview: Overview.render, how: How.render, lab: Lab.render, explorer: Explorer.render,
    results: Results.render, about: About.render };
  const rendered = new Set();

  function show(id) {
    const sec = SECTIONS.includes(id) ? id : "overview";
    SECTIONS.forEach(s => { U.$(`#s-${s}`).hidden = s !== sec; });
    U.$$(".tabs a").forEach(a => a.setAttribute("aria-current", a.getAttribute("href") === `#${sec}` ? "page" : "false"));
    if (!rendered.has(sec)) {
      try { RENDER[sec](); } catch (err) {
        console.error(err);
        U.$(`#s-${sec}`).innerHTML = `<div class="wrap"><div class="status-card"><h3>This section could not be displayed</h3>
          <p class="muted">${U.esc(err.message)}</p><p class="small">Rebuild the data with <code>uv run skin-research build-site</code>.</p></div></div>`;
      }
      rendered.add(sec);
    }
    U.hideTip();
    document.title = `${U.$(`.tabs a[href="#${sec}"]`).textContent.replace(/^\d+/, "").trim()} · Skin Lesion Image Quality & AI Reliability`;
  }

  function route() {
    const raw = location.hash.slice(1);
    if (raw === "tour" || raw.startsWith("tour")) {   // #tour or #tour3 opens the guided tour (for presenting)
      show("overview");
      Tour.open(Math.max(0, (parseInt(raw.slice(4), 10) || 1) - 1));
      return;
    }
    const [sec, sub] = raw.split("-");
    if (raw.startsWith("about-")) {           // in-page links inside About
      show("about");
      const t = document.getElementById(raw);
      if (t) t.scrollIntoView();
      return;
    }
    show(sec);
    if (!sub) window.scrollTo({ top: 0, behavior: "instant" });
  }

  window.addEventListener("hashchange", route);
  Tour.init();
  route();
  U.$("#generated").textContent = `Data built ${D.generated_utc.replace("T", " ").replace("+00:00", " UTC")} from experiment outputs.`;
})();

/* Developer self-test (see scripts/check_website.py): only runs with ?selftest in the URL. */
if (new URLSearchParams(location.search).has("selftest")) {
  const s = document.createElement("script");
  s.src = "js/selftest.js";
  document.body.appendChild(s);
}
