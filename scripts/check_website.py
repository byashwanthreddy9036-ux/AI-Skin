#!/usr/bin/env python3
"""Run the website's in-browser self-test in headless Chrome/Edge at several screen widths.

    uv run python scripts/check_website.py            # widths 1440 (desktop), 768 (tablet), 375 (phone)
    uv run python scripts/check_website.py --widths 375

Starts a local server for website/, opens index.html?selftest (or frame.html for widths below
~1000 px, because headless browsers cannot open very narrow windows), and prints the result:
navigation, every Image Lab combination, Explorer cards, Results controls, tour steps, layout
overflow and image loading. Exit code 1 if anything fails. Works on Linux, macOS and WSL (where
it uses the Windows browser).
"""

from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"
CANDIDATES = [
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge",
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    for c in CANDIDATES:
        path = shutil.which(c) or (c if Path(c).is_file() else None)
        if path:
            return path
    sys.exit("No Chrome/Chromium/Edge found. Install one, or open website/index.html?selftest by hand.")


def serve(port: int) -> http.server.ThreadingHTTPServer:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=str(SITE)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def run(browser: str, url: str) -> dict:
    out = subprocess.run([browser, "--headless=new", "--disable-gpu", "--window-size=1440,900",
                          "--virtual-time-budget=180000", "--dump-dom", url],
                         capture_output=True, text=True, timeout=600)
    m = re.search(r'<pre id="selftest-result">(.*?)</pre>', out.stdout, re.S)
    if not m:
        return {"passed": False, "errors": ["self-test produced no result (page failed to load?)"], "failures": []}
    return json.loads(html.unescape(m.group(1)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the website self-test in a headless browser.")
    ap.add_argument("--widths", type=int, nargs="+", default=[1440, 768, 375])
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    if not (SITE / "data" / "site-data.js").is_file():
        sys.exit("website/data/site-data.js missing: run `uv run skin-research build-site` first.")
    browser = find_browser()
    httpd = serve(args.port)
    ok = True
    try:
        for w in args.widths:
            url = (f"http://localhost:{args.port}/?selftest" if w >= 1000
                   else f"http://localhost:{args.port}/frame.html?selftest&w={w}")
            r = run(browser, url)
            ok &= bool(r.get("passed"))
            print(f"width {w:>5}px: {'PASS' if r.get('passed') else 'FAIL'}  checks={r.get('checks')} "
                  f"lab_combinations={r.get('lab_combinations')} tour_slides={r.get('tour_slides')} "
                  f"images={(r.get('images') or {}).get('checked')}")
            for key in ("failures", "errors", "overflow"):
                for item in (r.get(key) or [])[:10]:
                    print(f"    {key}: {item}")
            for img in ((r.get("images") or {}).get("failed") or [])[:10]:
                print(f"    missing image: {img}")
    finally:
        httpd.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
