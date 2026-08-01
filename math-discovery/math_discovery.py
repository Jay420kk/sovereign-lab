#!/usr/bin/env python3
"""math_discovery.py — a pocket-sized 'Ramanujan Machine'.

The Ramanujan Machine finds NEW mathematical identities by searching for
integer relations among constants computed to high precision (the PSLQ
algorithm). A relation a*A + b*B + c*C ~ 0 with small integers a,b,c is a
candidate identity that mathematicians then try to prove.

Here we do exactly that, on a laptop: compute a bank of constants exactly
(rational arithmetic, no float drift), then brute-force small-coefficient
relations. Any hit is flagged for proof.

Usage: python3 math_discovery.py [search_digits]
"""

import json
import math
import sys
import time
from fractions import Fraction
from itertools import combinations
from pathlib import Path

OUT_DIR = Path(__file__).parent
LOGDIR = OUT_DIR / "logs"


def atan_series(n, prec=80):
    """atan(1/n) as an exact Fraction, |term| bounded by 10^-prec."""
    x = Fraction(1, n)
    x2 = x * x
    total, term, k = x, x, 1
    while abs(term) > Fraction(1, 10 ** (prec + 1)):
        term *= -x2
        k += 2
        total += term / k
    return total


def build_constants(prec=80):
    """A bank of constants as exact Fractions. These are the 'letters' our
    identities are built from. Each value is truncated to prec digits, so
    'exact' relations below hold to the working precision.

    The bank deliberately includes the classic Machin-like arctangents so the
    search has real, verifiable identities to rediscover (pi = 16*atan(1/5)
    - 4*atan(1/239), pi/4 = atan(1/2)+atan(1/3), ...).
    """
    pi = 4 * (4 * atan_series(5, prec) - atan_series(239, prec))
    e = Fraction(0)
    # e = sum 1/k!  (exact, just many terms)
    k, term = 0, Fraction(1)
    while term > Fraction(1, 10 ** (prec + 1)):
        e += term
        k += 1
        term /= k

    import math as m
    scale = 10 ** prec
    sqrt2 = Fraction(m.isqrt(2 * scale * scale), scale)
    sqrt3 = Fraction(m.isqrt(3 * scale * scale), scale)
    sqrt5 = Fraction(m.isqrt(5 * scale * scale), scale)
    ln2 = Fraction(0)
    x = Fraction(1, 2)
    k, term = 1, x
    while abs(term) > Fraction(1, 10 ** (prec + 1)):
        ln2 += term / k
        k += 1
        term *= x

    return {
        "1": Fraction(1),
        "pi": pi,
        "pi/4": pi / 4,
        "pi^2": pi * pi,
        "e": e,
        "e^2": e * e,
        "sqrt2": sqrt2,
        "sqrt3": sqrt3,
        "sqrt5": sqrt5,
        "phi": (1 + sqrt5) / 2,
        "ln2": ln2,
        "atan(1/2)": atan_series(2, prec),
        "atan(1/3)": atan_series(3, prec),
        "atan(1/5)": atan_series(5, prec),
        "atan(1/7)": atan_series(7, prec),
        "atan(1/239)": atan_series(239, prec),
    }


def find_relations(consts, deadline=None):
    """Search small-integer relations among 2-3 constants, exact arithmetic:

      pairs:    a*X + b*Y == 0
      triples:  a*X + b*Y + c*Z == 0   (c solved exactly; accepted when integer)

    A fast float probe gates the expensive exact-Fraction verification, so the
    search stays cheap even at high precision. Exact comparisons catch
    relations that hold by construction (pi = 4*(pi/4)); float near-misses
    catch truncated-series identities (Machin's formula etc.) that hold to the
    working precision. Deadline-aware so the worker never overshoots.
    """
    names = list(consts)
    fvals = {n: float(v) for n, v in consts.items()}
    hits = []
    seen = set()

    def emit(rel, tag):
        rel = {n: c for n, c in rel.items() if c != 0}
        if len(rel) < 2:
            return
        g = 0
        for c in rel.values():
            g = math.gcd(g, abs(c))
        if g > 1:  # reduce to the primitive relation so scaled copies dedupe
            rel = {n: c // g for n, c in rel.items()}
        items = sorted(rel.items())
        if items[0][1] < 0:  # normalize sign so (a,b,c) and (-a,-b,-c) dedupe
            rel = {n: -c for n, c in rel.items()}
            items = sorted(rel.items())
        key = tuple(items)
        if key in seen:
            return
        seen.add(key)
        hits.append(" + ".join(f"{c}*{n}" for n, c in items) + f" == 0  ({tag})")

    # --- pairs: a*X + b*Y == 0 ---
    for i, j in combinations(range(len(names)), 2):
        A, B = consts[names[i]], consts[names[j]]
        for a in range(-8, 9):
            if a == 0:
                continue
            for b in range(-8, 9):
                if b == 0:
                    continue
                s = a * A + b * B
                if s == 0:
                    emit({names[i]: a, names[j]: b}, "exact")
                elif abs(float(s)) < 1e-12:
                    emit({names[i]: a, names[j]: b}, "near-miss")

    # --- triples: solve c = -(a*A + b*B)/C, accept small integer c ---
    for i, j, k in combinations(range(len(names)), 3):
        A, B, C = consts[names[i]], consts[names[j]], consts[names[k]]
        fA, fB, fC = fvals[names[i]], fvals[names[j]], fvals[names[k]]
        for a in range(-17, 18):
            if a == 0:
                continue
            for b in range(-17, 18):
                if b == 0:
                    continue
                fc = -(a * fA + b * fB) / fC   # cheap float probe
                n = round(fc)
                if n == 0 or abs(n) > 30 or abs(fc - n) > 1e-6:
                    continue
                c = -(a * A + b * B) / C       # exact verification
                if c.denominator == 1:
                    cc = c.numerator
                    if cc != 0 and abs(cc) <= 30:
                        emit({names[i]: a, names[j]: b, names[k]: cc}, "exact")
                else:
                    cf = float(c)
                    n = round(cf)
                    if n != 0 and abs(n) <= 30 and abs(cf - n) < 1e-12:
                        tag = "exact" if cf == n else f"near-miss {abs(cf-n):.1e}"
                        emit({names[i]: a, names[j]: b, names[k]: n}, tag)
        if deadline is not None and time.time() > deadline:
            hits.append("(search stopped: time budget exceeded)")
            return hits
    return hits


def main(prec=80, deadline=None):
    LOGDIR.mkdir(exist_ok=True)
    t0 = time.time()
    consts = build_constants(prec)
    print("constants bank:")
    for k, v in consts.items():
        print(f"  {k:6s} = {float(v):.15f}")

    hits = find_relations(consts, deadline=deadline)
    print(f"\n== integer relations found (in {time.time()-t0:.1f}s) ==")
    for h in hits:
        print(" ", h)
    if not hits:
        print("  (none — expected for independent constants; try more constants / wider coefs)")

    with open(LOGDIR / "math_findings.json", "w") as f:
        json.dump({"constants": {k: float(v) for k, v in consts.items()}, "hits": hits}, f, indent=2)
    print(f"\nlogged -> {LOGDIR/'math_findings.json'}")


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    main(p)
