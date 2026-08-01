#!/usr/bin/env python3
"""Unit tests for remote-worker/sovereign-lab/worker.py (stdlib unittest).

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

import worker  # noqa: E402


class TestLoadSeed(unittest.TestCase):
    def _patch_root(self, root):
        self._old_root = worker.ROOT
        worker.ROOT = Path(root)

    def tearDown(self):
        if hasattr(self, "_old_root"):
            worker.ROOT = self._old_root

    def test_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self._patch_root(d)
            self.assertIsNone(worker.load_seed())

    def test_valid_seed_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / "logs"
            logs.mkdir()
            (logs / "evolve_best.json").write_text(json.dumps({
                "level": 3, "layers": [6, 14, 2],
                "w1": [[0.1] * 6 for _ in range(14)],
                "w2": [[0.1] * 14 for _ in range(2)],
            }))
            self._patch_root(d)
            seed = worker.load_seed()
            self.assertIsNotNone(seed)
            self.assertEqual(seed["level"], 3)

    def test_malformed_json_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / "logs"
            logs.mkdir()
            (logs / "evolve_best.json").write_text("not json{")
            self._patch_root(d)
            self.assertIsNone(worker.load_seed())

    def test_seed_without_w1_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            logs = Path(d) / "logs"
            logs.mkdir()
            (logs / "evolve_best.json").write_text(json.dumps({"level": 1}))
            self._patch_root(d)
            self.assertIsNone(worker.load_seed())


class TestGensScaling(unittest.TestCase):
    def test_gens_scales_with_cores(self):
        # over-requested cap scales with workers: 4 cores -> 4x the 1-core cap
        g1 = worker.planned_gens(300, 1)
        g4 = worker.planned_gens(300, 4)
        self.assertAlmostEqual(g4 / g1, 4.0, places=2)

    def test_gens_scales_with_budget(self):
        # bigger budget -> proportionally bigger cap
        g100 = worker.planned_gens(100, 4)
        g300 = worker.planned_gens(300, 4)
        self.assertGreater(g300, g100)
        self.assertAlmostEqual(g300 / g100, 3.0, places=1)

    def test_gens_over_requests_deadline_binds(self):
        # the cap must be far above what could run in budget: the deadline,
        # not the cap, is the real guard (so under-requesting never wastes
        # budget by finishing evolve early)
        self.assertGreater(worker.planned_gens(300, 4), 10000)

    def test_gens_min_floor(self):
        # sub-minute budget still gets the 50-gen floor
        self.assertEqual(worker.planned_gens(0.1, 4), 50)


if __name__ == "__main__":
    unittest.main()
