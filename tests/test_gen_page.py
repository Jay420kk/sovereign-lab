#!/usr/bin/env python3
"""Unit tests for remote-worker/sovereign-lab/gen_page.py (stdlib unittest).

Run from repo root:  python3 -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = ROOT / "remote-worker" / "sovereign-lab"
sys.path.insert(0, str(WORKER_DIR))

import gen_page  # noqa: E402


def write_fixture(root):
    logs = Path(root) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "island_manifest.json").write_text(json.dumps([
        {"ts": "2026-08-01T05:00:00Z", "peak": 0.97, "gens": 50, "level": 0,
         "seed_level": None, "ascensions": 0, "math_hits": 0},
        {"ts": "2026-08-01T11:00:00Z", "peak": 0.996, "gens": 480, "level": 4,
         "seed_level": 3, "ascensions": 4, "math_hits": 5},
    ]))
    (logs / "evolve_scores.json").write_text(json.dumps({
        "meta": {"ts": "2026-08-01T11:00:00Z"}, "gens": 480, "peak": 0.996,
        "scores": [0.5] * 480, "final_level": 4, "seed_level": 3,
        "ascensions": [{"level": i, "gen": 100 + i * 5, "peak": 0.98} for i in range(4)],
    }))
    (logs / "math_findings.json").write_text(json.dumps({
        "constants": {"pi": 3.14}, "hits": ["4*atan(1/239) + -16*atan(1/5) + 1*pi == 0  (exact)"],
    }))
    (logs / "status.json").write_text(json.dumps({
        "journal": [{"ts": "2026-08-01T10:00:00Z", "agent": "research",
                     "question": "q?", "response": "a"}],
        "notes": "some notes",
    }))
    return logs


class TestGenPage(unittest.TestCase):
    def test_renders_with_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            logs = write_fixture(d)
            old_root, old_logs = gen_page.ROOT, gen_page.LOGS
            gen_page.ROOT, gen_page.LOGS = Path(d), logs
            try:
                gen_page.main()
                html = (Path(d) / "docs" / "index.html").read_text()
            finally:
                gen_page.ROOT, gen_page.LOGS = old_root, old_logs

            self.assertIn("Cloud islands", html)
            self.assertIn("0.996", html)          # peak in table
            self.assertIn("L3", html)             # seeded column
            self.assertIn("All-time best peak", html)
            self.assertIn("atan(1/239)", html)   # math hit rendered
            self.assertIn("<svg", html)           # sparkline present

    def test_renders_empty_lab(self):
        with tempfile.TemporaryDirectory() as d:
            old_root, old_logs = gen_page.ROOT, gen_page.LOGS
            gen_page.ROOT, gen_page.LOGS = Path(d), Path(d) / "logs"
            try:
                gen_page.main()  # no logs at all — must not raise
                html = (Path(d) / "docs" / "index.html").read_text()
            finally:
                gen_page.ROOT, gen_page.LOGS = old_root, old_logs
            self.assertIn("no islands yet", html)


if __name__ == "__main__":
    unittest.main()
