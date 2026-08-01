#!/usr/bin/env python3
"""Unit tests for evolve_mind.py — focuses on the seeding / escalation paths
that regressed before. Uses a temp LOGDIR so real lab logs are never touched.

Run from repo root:  python3 -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evolve"))

import evolve_mind  # noqa: E402


def run_tiny(**kwargs):
    """run() with a tiny population/gens, forced to the deterministic SERIAL
    scoring path (no multiprocessing pool) so unit tests stay fast and
    reproducible. The parallel path is smoke-tested separately below."""
    with tempfile.TemporaryDirectory() as d:
        old_logdir = evolve_mind.LOGDIR
        old_pool = evolve_mind._make_pool
        evolve_mind.LOGDIR = Path(d)
        evolve_mind._make_pool = lambda: None  # force serial
        try:
            evolve_mind.run(generations=3, pop_size=8, **kwargs)
            return json.loads((Path(d) / "evolve_scores.json").read_text())
        finally:
            evolve_mind.LOGDIR = old_logdir
            evolve_mind._make_pool = old_pool


class TestFreshRun(unittest.TestCase):
    def test_fresh_run_shape(self):
        out = run_tiny()
        self.assertEqual(out["seed_level"], None)
        self.assertEqual(len(out["scores"]), 3)
        self.assertIn("final_level", out)
        self.assertIn("ascensions", out)

    def test_fresh_run_no_seed_best(self):
        with tempfile.TemporaryDirectory() as d:
            old_logdir = evolve_mind.LOGDIR
            evolve_mind.LOGDIR = Path(d)
            try:
                evolve_mind.run(generations=2, pop_size=8)
                best = json.loads((Path(d) / "evolve_best.json").read_text())
                self.assertIn("w1", best)
                self.assertIn("w2", best)
            finally:
                evolve_mind.LOGDIR = old_logdir


class TestSeededRun(unittest.TestCase):
    def _seed(self, level, old_format=False):
        if old_format:
            # a brain saved by the pre-fix LEVELS (input 18, output 8)
            return {"level": level, "layers": [18, 18, 8],
                    "w1": [[0.1] * 18 for _ in range(18)],
                    "w2": [[0.1] * 18 for _ in range(8)]}
        s = evolve_mind.make_brain(0, (6, 6, 2))
        s["level"] = level
        return s

    def test_seed_level_zero(self):
        out = run_tiny(seed=self._seed(0))
        self.assertEqual(out["seed_level"], 0)

    def test_seed_level_two(self):
        out = run_tiny(seed=self._seed(2))
        self.assertEqual(out["seed_level"], 2)
        self.assertEqual(out["final_level"], 2)

    def test_seed_max_level(self):
        out = run_tiny(seed=self._seed(evolve_mind.MAX_LEVEL))
        self.assertEqual(out["seed_level"], evolve_mind.MAX_LEVEL)

    def test_seed_old_format_no_crash(self):
        # old-format 18-wide brain must embed (min-dims copy) without crashing
        out = run_tiny(seed=self._seed(4, old_format=True))
        self.assertEqual(out["seed_level"], 4)
        self.assertEqual(len(out["scores"]), 3)

    def test_seed_null_level(self):
        s = self._seed(0)
        s["level"] = None
        out = run_tiny(seed=s)  # must not raise; treated as level 0
        self.assertEqual(out["seed_level"], None)


class TestEscalation(unittest.TestCase):
    def test_levels_fixed_widths(self):
        # input width (6 sensors) and output width (dx,dy=2) are fixed
        for lvl in evolve_mind.LEVELS:
            self.assertEqual(lvl["layers"][0], 6)
            self.assertEqual(lvl["layers"][2], 2)


class TestParallelPath(unittest.TestCase):
    def test_parallel_scoring_smoke(self):
        """Real multiprocessing pool: must produce a valid best brain without
        hanging. Uses the default _make_pool (capped at 4 workers)."""
        with tempfile.TemporaryDirectory() as d:
            old_logdir = evolve_mind.LOGDIR
            evolve_mind.LOGDIR = Path(d)
            try:
                evolve_mind.run(generations=1, pop_size=8)
                out = json.loads((Path(d) / "evolve_scores.json").read_text())
                self.assertEqual(len(out["scores"]), 1)
                best = json.loads((Path(d) / "evolve_best.json").read_text())
                self.assertIn("w1", best)
            finally:
                evolve_mind.LOGDIR = old_logdir


if __name__ == "__main__":
    unittest.main()
