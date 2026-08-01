#!/usr/bin/env python3
"""gen_page.py — build docs/index.html (phone-accessible lab dashboard).

Reads committed results + laptop status and renders a single static page:
  - cloud island manifest (evolve peaks, gens, math hits)
  - latest island score curve
  - latest math findings
  - laptop lab status (journal tail, notes, local evolve)

Pure stdlib. Run on the worker after each island and on the laptop after
each sync.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


def read_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    islands = read_json(LOGS / "island_manifest.json", []) or []
    latest = read_json(LOGS / "evolve_scores.json")
    mathf = read_json(LOGS / "math_findings.json")
    status = read_json(LOGS / "status.json")

    sections = []

    rows = "".join(
        "<tr><td>{ts}</td><td>{peak}</td><td>{gens}</td><td>{lvl}</td><td>{asc}</td><td>{hits}</td></tr>".format(
            ts=esc(i.get("ts", "")), peak=i.get("peak", "?"),
            gens=i.get("gens", "?"), lvl=i.get("level", "?"),
            asc=i.get("ascensions", 0), hits=i.get("math_hits", "?"))
        for i in islands[-30:][::-1])
    sections.append(
        "<h2>Cloud islands (past 30 of {n})</h2>".format(n=len(islands))
        + "<table><tr><th>time (UTC)</th><th>peak</th><th>gens</th><th>level</th><th>ascensions</th><th>math hits</th></tr>"
        + rows + "</table>" if rows else "<h2>Cloud islands</h2><p>no islands yet</p>")

    if latest and latest.get("scores"):
        scores = latest["scores"]
        final_level = latest.get("final_level", 0)
        asc = latest.get("ascensions", [])
        step = max(1, len(scores) // 120)
        pts = scores[::step]
        w, h = 800, 180
        mx = max(pts) or 1
        path = " ".join(
            "L{:.0f} {:.0f}".format(i / len(pts) * w, h - pts[i] / mx * h)
            for i in range(len(pts)))
        asc_text = ""
        if asc:
            asc_text = " &middot; ascended {}x (now L{})".format(len(asc), final_level)
        sections.append(
            "<h2>Latest island curve (peak {}{})</h2>".format(latest.get("peak"), asc_text)
            + "<svg viewBox='0 0 {0} {1}' width='100%'><path d='M0 {1} {2}' fill='none' stroke='#7fb3d5'/></svg>".format(w, h, path))

    if mathf:
        hits = mathf.get("hits") or []
        hs = "<br>".join(esc(str(h)) for h in hits) if hits else "none (expected)"
        sections.append("<h2>Latest math findings</h2><p>{}</p>".format(hs))

    if status:
        jr = "".join(
            "<div class='j'><b>[{ts} {agent}]</b> {q}<br>{a}</div>".format(
                ts=esc(e.get("ts", ""))[:16], agent=esc(e.get("agent", "?")),
                q=esc(e.get("question", ""))[:140],
                a=esc(e.get("response", ""))[:400])
            for e in (status.get("journal") or []))
        notes = esc(status.get("notes", ""))[:800]
        sections.append(
            "<h2>Laptop lab</h2><p>evolve gen {g} peak {p}</p>".format(
                g=status.get("gen", "?"), p=status.get("peak", "?"))
            + jr
            + "<details><summary>research notes tail</summary><p>{}</p></details>".format(notes))

    html = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sovereign Lab</title><style>
body {{ background: #101418; color: #d8dee9; font-family: monospace; max-width: 860px; margin: auto; padding: 1em }}
h1 {{ color: #7fb3d5 }}
h2 {{ color: #b48ead; border-bottom: 1px solid #1c232b; padding-bottom: 4px }}
table {{ width: 100%; border-collapse: collapse }}
td, th {{ border: 1px solid #1c232b; padding: 4px; text-align: left }}
.j {{ margin: 8px 0; padding: 6px; background: #161c22; border-radius: 4px }}
a {{ color: #7fb3d5 }}
summary {{ cursor: pointer }}
</style></head><body>
<h1>Sovereign Lab</h1>
<p>generated {ts} UTC &middot; <a href="https://github.com/Jay420kk/sovereign-lab">repo</a></p>
{sections}
</body></html>""".format(ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), sections="\n".join(sections))

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html)
    print("[page] docs/index.html written")


if __name__ == "__main__":
    main()
