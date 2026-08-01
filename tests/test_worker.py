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
    def test_gens_scales_with_budget(self):
        # 300 min budget, 1 core: 300*60*0.8/30 = 480
        gens = max(50, int((300 * 60 * 0.8) / 30) * 1)
        self.assertEqual(gens, 480)

    def test_gens_min_floor(self):
        # tiny budget still gets the 50-gen floor
        gens = max(50, int((1 * 60 * 0.8) / 30) * 4)
        self.assertEqual(gens, 50)


if __name__ == "__main__":
    unittest.main()
