#!/usr/bin/env python3
"""evolve_mind.py — evolve a tiny neural mind from raw noise.

No backprop, no gradients, no human-designed task priors beyond the
physics of the arena. Just: population -> mutate -> score -> select.

The task: a 2D agent with a fixed 6-6-2 network must HOME to food at
(5,5). Random brains wander and score ~0. Only a brain that actually
learns to steer toward the target can score high, so the learning curve
IS the visible emergence of intelligence.

Scoring is a logistic of (distance food) — smooth selection pressure.

Usage: python3 evolve_mind.py [generations]
"""

import json
import math
import random
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent
LOGDIR = OUT_DIR / "logs"


def make_brain(seed=None):
    rng = random.Random(seed)
    return {
        "w1": [[rng.uniform(-1.5, 1.5) for _ in range(6)] for _ in range(6)],
        "w2": [[rng.uniform(-1.5, 1.5) for _ in range(6)] for _ in range(2)],
    }


def brain_fwd(b, x):
    h = [math.tanh(sum(b["w1"][j][i] * x[i] for i in range(6))) for j in range(6)]
    return [math.tanh(sum(b["w2"][k][j] * h[j] for j in range(6))) for k in range(2)]


def mutate(b, rate=0.15, sigma=0.4):
    nb = make_brain()
    for layer in ("w1", "w2"):
        for r in range(len(nb[layer])):
            for c in range(len(nb[layer][r])):
                nb[layer][r][c] = b[layer][r][c]
                if random.random() < rate:
                    nb[layer][r][c] += random.gauss(0, sigma)
    return nb


def score_brain(b, steps=300, trials=8):
    total = 0.0
    for _ in range(trials):
        fx = random.uniform(-4, 4)
        fy = random.uniform(-4, 4)
        x, y = 0.0, 0.0
        min_dist = math.hypot(x - fx, y - fy)
        for s in range(steps):
            sens = [x / 6, y / 6, fx / 6, fy / 6, math.sin(0.05 * s), math.cos(0.05 * s)]
            dx, dy = brain_fwd(b, sens)
            x += 0.25 * dx
            y += 0.25 * dy
            x = max(-6.0, min(6.0, x))
            y = max(-6.0, min(6.0, y))
            d = math.hypot(x - fx, y - fy)
            min_dist = min(min_dist, d)
        total += 1.0 / (1.0 + min_dist / 1.5)
    return total / trials


def run(generations=200, pop_size=64):
    LOGDIR.mkdir(exist_ok=True)
    pop = [make_brain(i) for i in range(pop_size)]
    best_scores = []
    t0 = time.time()

    for gen in range(generations):
        scored = sorted(((score_brain(b), b) for b in pop), key=lambda p: p[0], reverse=True)
        best_score, best = scored[0]
        best_scores.append(best_score)
        if gen % 10 == 0 or gen == generations - 1:
            print(f"gen {gen:4d}  best={best_score:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

        nxt = [best]
        while len(nxt) < pop_size:
            parent = random.choice(scored[:16])[1]
            nxt.append(mutate(parent, rate=0.12, sigma=max(0.02, 0.4 / (1 + gen / 60))))
        pop = nxt

    with open(LOGDIR / "evolve_scores.json", "w") as f:
        json.dump(best_scores, f)
    with open(LOGDIR / "evolve_best.json", "w") as f:
        json.dump(best, f)
    print(f"done. peak best={max(best_scores):.4f}  -> {LOGDIR/'evolve_scores.json'}")
    return best_scores


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(n)
