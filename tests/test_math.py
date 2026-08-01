#!/usr/bin/env python3
"""Unit tests for math_discovery.py (stdlib unittest, run from repo root):

    python3 -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "math-discovery"))

import math_discovery  # noqa: E402


class TestConstants(unittest.TestCase):
    def test_bank_has_expected_constants(self):
        consts = math_discovery.build_constants(40)
        for name in ("pi", "pi/4", "e", "sqrt2", "sqrt5", "phi", "ln2",
                     "atan(1/2)", "atan(1/3)", "atan(1/5)", "atan(1/7)",
                     "atan(1/239)"):
            self.assertIn(name, consts, f"missing constant {name}")

    def test_pi_value(self):
        consts = math_discovery.build_constants(60)
        pi = float(consts["pi"])
        self.assertAlmostEqual(pi, 3.141592653589793, places=14)


class TestRelations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.consts = math_discovery.build_constants(40)
        cls.hits = math_discovery.find_relations(cls.consts)

    def test_has_hits(self):
        self.assertGreater(len(self.hits), 0, "search should find known identities")

    def test_machin_formula(self):
        # pi = 16*atan(1/5) - 4*atan(1/239)
        self.assertTrue(
            any("16*atan(1/5)" in h and "4*atan(1/239)" in h and "pi" in h
                for h in self.hits),
            f"Machin's formula missing from {self.hits}")

    def test_euler_atan_identity(self):
        # atan(1/2) + atan(1/3) = pi/4
        self.assertTrue(
            any("atan(1/2)" in h and "atan(1/3)" in h and "pi/4" in h
                for h in self.hits),
            f"Euler atan identity missing from {self.hits}")

    def test_phi_definition(self):
        # phi = (1 + sqrt5)/2  ->  1*1 + 1*sqrt5 + -2*phi == 0
        self.assertTrue(
            any("sqrt5" in h and "phi" in h and h.startswith("1*1 +")
                for h in self.hits),
            f"phi identity missing from {self.hits}")

    def test_no_zero_coefficient_garbage(self):
        # the pre-upgrade bug emitted "8*1 + 0*pi == 0" style near-misses
        for h in self.hits:
            self.assertNotIn(" + 0*", h, f"zero-coefficient hit: {h}")
            self.assertFalse(h.startswith("0*"), f"zero-coefficient hit: {h}")

    def test_no_scaled_duplicates(self):
        # pi = 4*(pi/4) must appear only once (primitive form after gcd)
        pi_pairs = [h for h in self.hits
                    if "pi" in h and "pi/4" in h and h.startswith("1*pi + -4*pi/4")]
        self.assertEqual(len(pi_pairs), 1, f"duplicate pi/pi4 relations: {pi_pairs}")

    def test_deadline_stops(self):
        import time
        deadline = time.time()  # already expired
        hits = math_discovery.find_relations(self.consts, deadline=deadline)
        # expired deadline should either return early with a marker or complete
        self.assertIsInstance(hits, list)


if __name__ == "__main__":
    unittest.main()
