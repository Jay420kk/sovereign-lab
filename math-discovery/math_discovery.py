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
import sys
from fractions import Fraction
from itertools import product
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
    identities are built from."""
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
    ln2 = Fraction(0)
    x = Fraction(1, 2)
    k, term = 1, x
    while abs(term) > Fraction(1, 10 ** (prec + 1)):
        ln2 += term / k
        k += 1
        term *= x

    return {
        "pi": pi,
        "pi2": pi * pi,
        "e": e,
        "e2": e * e,
        "sqrt2": sqrt2,
        "ln2": ln2,
    }


def find_relations(consts, coef_range=(-4, 5)):
    """Search a*c_i + b*c_j + c*1 == 0 for small integer a,b,c."""
    names = list(consts)
    hits = []
    for i, j in product(range(len(names)), repeat=2):
        if i >= j:
            continue
        A, B = consts[names[i]], consts[names[j]]
        for a, b in product(range(*coef_range), repeat=2):
            if a == 0 and b == 0:
                continue
            if a * A + b * B == 0:
                hits.append(f"{a}*{names[i]} + {b}*{names[j]} == 0  (exact proportionality)")
                continue
            c = -(a * A + b * B)
            # want c to be (approximately) an integer
            c_float = float(c)
            nearest = round(c_float)
            if abs(c_float - nearest) < 1e-14 and abs(nearest) <= 10:
                if a * A + b * B + nearest == 0:
                    hits.append(f"{a}*{names[i]} + {b}*{names[j]} + {nearest} == 0  (exact)")
                else:
                    hits.append(f"{a}*{names[i]} + {b}*{names[j]} ~ {nearest}  (near-miss {abs(c_float-nearest):.1e})")
    return hits


def main(prec=80):
    LOGDIR.mkdir(exist_ok=True)
    consts = build_constants(prec)
    print("constants bank:")
    for k, v in consts.items():
        print(f"  {k:6s} = {float(v):.15f}")

    hits = find_relations(consts)
    print("\n== integer relations found ==")
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
