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


def atanh_series(n, prec=80):
    """atanh(1/n) as an exact Fraction. ln((n+1)/(n-1)) = 2*atanh(1/n)."""
    x = Fraction(1, n)
    x2 = x * x
    total, term, k = x, x, 1
    while abs(term) > Fraction(1, 10 ** (prec + 1)):
        term *= x2
        k += 2
        total += term / k
    return total


def exp_frac(x, prec=80):
    """exp(x) for Fraction x as an exact Fraction."""
    total, term, k = Fraction(1), Fraction(1), 0
    while abs(term) > Fraction(1, 10 ** (prec + 1)):
        k += 1
        term *= x
        term /= k
        total += term
    return total


def zeta3_series(prec=80):
    """zeta(3) via the fast-convergent Apery series.

    zeta(3) = 5/2 * sum_{k>=1} (-1)^(k-1) / (k^3 * C(2k,k))
    Each term ratio is ~1/8, so ~90 terms give 120 digits — practical as
    exact fractions. zeta(3) is famously NOT known to be rational, so any
    small-integer relation involving it would be a genuine discovery.
    """
    total = Fraction(0)
    k = 1
    while True:
        # C(2k,k) as an exact fraction
        binom = Fraction(1)
        for j in range(1, k + 1):
            binom = binom * (2 * k - j + 1) / j
        term = Fraction(1, k * k * k) / binom
        if k % 2 == 1:
            total += term
        else:
            total -= term
        if term < Fraction(1, 10 ** (prec + 2)):
            break
        k += 1
    return Fraction(5, 2) * total


def build_constants(prec=80):
    """A bank of constants as exact Fractions. These are the 'letters' our
    identities are built from. Each value is truncated to prec digits, so
    'exact' relations below hold to the working precision.

    The bank deliberately includes the classic Machin-like arctangents so the
    search has real, verifiable identities to rediscover (pi = 16*atan(1/5)
    - 4*atan(1/239), pi/4 = atan(1/2)+atan(1/3), ...). It ALSO includes
    constants that are suspected to be algebraically independent (zeta(3),
    e^pi, 2^sqrt2, ln3, ln5, ...) — a small-integer relation among THOSE would
    be a genuinely new mathematical result, not a rediscovery.
    """
    pi = 4 * (4 * atan_series(5, prec) - atan_series(239, prec))
    e = exp_frac(Fraction(1), prec)

    scale = 10 ** prec
    def isqrt_frac(n):
        return Fraction(int(math.isqrt(n * scale * scale)), scale)

    sqrt2 = isqrt_frac(2)
    sqrt3 = isqrt_frac(3)
    sqrt5 = isqrt_frac(5)
    sqrt6 = isqrt_frac(6)
    sqrt7 = isqrt_frac(7)
    sqrt10 = isqrt_frac(10)

    ln2 = 2 * atanh_series(3, prec)          # atanh(1/3) = 1/2 ln(2)
    ln3 = 2 * atanh_series(2, prec)          # atanh(1/2) = 1/2 ln(3)
    ln5 = 2 * ln2 + 2 * atanh_series(9, prec)  # ln(5) = 2 ln2 + 2 atanh(1/9)

    e_pi = exp_frac(pi, prec)                # Gelfond's constant (transcendental)
    two_sqrt2 = exp_frac(sqrt2 * ln2, prec)  # Gelfond-Schneider 2^sqrt2

    return {
        "1": Fraction(1),
        "pi": pi,
        "pi/4": pi / 4,
        "pi^2": pi * pi,
        "pi^3": pi * pi * pi,
        "e": e,
        "e^2": e * e,
        "e^pi": e_pi,
        "2^sqrt2": two_sqrt2,
        "zeta(3)": zeta3_series(prec),
        "sqrt2": sqrt2,
        "sqrt3": sqrt3,
        "sqrt5": sqrt5,
        "sqrt6": sqrt6,
        "sqrt7": sqrt7,
        "sqrt10": sqrt10,
        "phi": (1 + sqrt5) / 2,
        "phi^2": ((1 + sqrt5) / 2) ** 2,
        "ln2": ln2,
        "ln3": ln3,
        "ln5": ln5,
        "atan(1/2)": atan_series(2, prec),
        "atan(1/3)": atan_series(3, prec),
        "atan(1/5)": atan_series(5, prec),
        "atan(1/7)": atan_series(7, prec),
        "atan(1/8)": atan_series(8, prec),
        "atan(1/9)": atan_series(9, prec),
        "atan(1/11)": atan_series(11, prec),
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
    fvals = [float(v) for n, v in consts.items()]
    hits = []
    seen = set()
    n_c = len(names)

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
    for i, j in combinations(range(n_c), 2):
        A, B = consts[names[i]], consts[names[j]]
        fA, fB = fvals[i], fvals[j]
        for a in range(-12, 13):
            if a == 0:
                continue
            for b in range(-12, 13):
                if b == 0:
                    continue
                # float probe first; exact only on near-integer ratios
                if abs((a * fA + b * fB) / max(abs(fA), abs(fB))) > 1e-15:
                    continue
                s = a * A + b * B
                if s == 0:
                    emit({names[i]: a, names[j]: b}, "exact")
        if deadline is not None and time.time() > deadline:
            hits.append("(search stopped: time budget exceeded)")
            return hits

    # --- triples: solve c = -(a*A + b*B)/C, accept small integer c ---
    for i, j, k in combinations(range(n_c), 3):
        A, B, C = consts[names[i]], consts[names[j]], consts[names[k]]
        fA, fB, fC = fvals[i], fvals[j], fvals[k]
        for a in range(-20, 21):
            if a == 0:
                continue
            for b in range(-20, 21):
                if b == 0:
                    continue
                fc = -(a * fA + b * fB) / fC   # cheap float probe
                n = round(fc)
                if n == 0 or abs(n) > 40 or abs(fc - n) > 1e-9:
                    continue
                c = -(a * A + b * B) / C       # exact verification
                if c.denominator == 1:
                    cc = c.numerator
                    if cc != 0 and abs(cc) <= 40:
                        emit({names[i]: a, names[j]: b, names[k]: cc}, "exact")
                else:
                    cf = float(c)
                    n = round(cf)
                    if n != 0 and abs(n) <= 40 and abs(cf - n) < 1e-15:
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
