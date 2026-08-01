#!/usr/bin/env python3
"""evolve_mind.py — open-ended evolutionary mind.

No backprop, no gradients, no human-designed task priors beyond the
physics of the arena. Just: population -> mutate -> score -> select.

The task: a 2D agent with a fixed small network must HOME to food.
Scoring is 1/(1 + dist/dist_scale) — smooth selection pressure, max 1.0.

OPEN-ENDED ESCALATION: when the population sustains best >= threshold
for N consecutive generations, the task ASCENDS a level: bigger arena,
longer runs, a larger brain (old weights embedded top-left, rest random).
The learned structure survives; the brain grows. This keeps evolution
under pressure past any single plateau.

Sensors are arena-normalized (x/clamp, y/clamp), so scores stay
comparable across levels.

Usage: python3 evolve_mind.py [generations]
"""

import json
import math
import random
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent
LOGDIR = OUT_DIR / "logs"

# input width (6 sensors) and output width (2D dx,dy movement) are FIXED;
# escalation grows only the hidden layer. Growing input/output widths would
# break score_brain (it feeds 6 sensors and unpacks exactly 2 outputs).
LEVELS = [
    dict(layers=[6, 6, 2], food_r=4.0, steps=300, trials=8, clamp=6.0, dist_scale=1.5),
    dict(layers=[6, 8, 2], food_r=8.0, steps=400, trials=8, clamp=12.0, dist_scale=3.0),
    dict(layers=[6, 10, 2], food_r=16.0, steps=500, trials=10, clamp=24.0, dist_scale=6.0),
    dict(layers=[6, 14, 2], food_r=32.0, steps=600, trials=12, clamp=48.0, dist_scale=12.0),
    dict(layers=[6, 18, 2], food_r=64.0, steps=700, trials=14, clamp=96.0, dist_scale=24.0),
]
MAX_LEVEL = len(LEVELS) - 1


def make_brain(seed=None, sizes=(6, 6, 2)):
    rng = random.Random(seed)
    return {
        "w1": [[rng.uniform(-1.5, 1.5) for _ in range(sizes[0])] for _ in range(sizes[1])],
        "w2": [[rng.uniform(-1.5, 1.5) for _ in range(sizes[1])] for _ in range(sizes[2])],
    }


def brain_fwd(b, x):
    w1, w2 = b["w1"], b["w2"]
    # pad sensors to the brain's input width so grown/legacy brains (wider
    # w1) never crash — extra inputs just see 0, so old weights stay valid
    if len(x) < len(w1[0]):
        x = x + [0.0] * (len(w1[0]) - len(x))
    h = [math.tanh(sum(w1[j][i] * x[i] for i in range(len(w1[0])))) for j in range(len(w1))]
    return [math.tanh(sum(w2[k][j] * h[j] for j in range(len(h)))) for k in range(len(w2))]


def mutate(b, rate=0.15, sigma=0.4):
    nb = {k: [row[:] for row in v] for k, v in b.items()}
    for layer in ("w1", "w2"):
        for r in range(len(nb[layer])):
            for c in range(len(nb[layer][r])):
                if random.random() < rate:
                    nb[layer][r][c] += random.gauss(0, sigma)
    return nb


def grow_brain(b, new_sizes):
    """Carry the old brain's weights into a larger net (top-left corner)."""
    nb = make_brain(sizes=new_sizes)
    for r in range(min(len(b["w1"]), len(nb["w1"]))):
        for c in range(min(len(b["w1"][0]), len(nb["w1"][0]))):
            nb["w1"][r][c] = b["w1"][r][c]
    for r in range(min(len(b["w2"]), len(nb["w2"]))):
        for c in range(min(len(b["w2"][0]), len(nb["w2"][0]))):
            nb["w2"][r][c] = b["w2"][r][c]
    return nb


def score_brain(b, task, rng):
    total = 0.0
    for _ in range(task["trials"]):
        fx = rng.uniform(-task["food_r"], task["food_r"])
        fy = rng.uniform(-task["food_r"], task["food_r"])
        x, y = 0.0, 0.0
        min_dist = math.hypot(x - fx, y - fy)
        for s in range(task["steps"]):
            sens = [x / task["clamp"], y / task["clamp"],
                    fx / task["clamp"], fy / task["clamp"],
                    math.sin(0.05 * s), math.cos(0.05 * s)]
            dx, dy = brain_fwd(b, sens)
            x += 0.25 * dx
            y += 0.25 * dy
            x = max(-task["clamp"], min(task["clamp"], x))
            y = max(-task["clamp"], min(task["clamp"], y))
            d = math.hypot(x - fx, y - fy)
            min_dist = min(min_dist, d)
        total += 1.0 / (1.0 + min_dist / task["dist_scale"])
    return total / task["trials"]


def run(generations=200, pop_size=64, ascend_threshold=0.98, ascend_streak_req=5,
        deadline=None, seed=None):
    """Evolve; ASCEND levels when the task is solved. Returns best-score list.

    deadline: unix ts — the loop stops cleanly at the budget, so callers
    (worker islands) always get results even if escalation slowed things down.

    seed: dict {"level", "w1", "w2", ...} — the best brain from a previous
    island (cross-island inheritance). The population starts at the seed's
    level with the old weights embedded top-left, so evolution accumulates
    across islands instead of restarting from scratch every run.
    """
    LOGDIR.mkdir(exist_ok=True)
    rng = random.Random()
    if seed and seed.get("w1") and seed.get("w2"):
        level = min(max(int(seed.get("level") or 0), 0), MAX_LEVEL)
        task = LEVELS[level]
        pop = [grow_brain(seed, task["layers"])] + \
              [make_brain(None, task["layers"]) for _ in range(pop_size - 1)]
        print(f"seeded population at level {level} (layers={task['layers']}) "
              f"from previous island's best", flush=True)
    else:
        level = 0
        task = LEVELS[level]
        pop = [make_brain(i, task["layers"]) for i in range(pop_size)]
    best_scores, ascensions = [], []
    streak = 0
    t0 = time.time()

    for gen in range(generations):
        if deadline is not None and time.time() > deadline:
            print(f"budget exhausted at gen {gen}", flush=True)
            break
        scored = sorted(((score_brain(b, task, rng), b) for b in pop),
                        key=lambda p: p[0], reverse=True)
        best_score, best = scored[0]
        best_scores.append(best_score)
        tag = ""
        if best_score >= ascend_threshold:
            streak += 1
            if streak >= ascend_streak_req and level < MAX_LEVEL:
                ascensions.append({"level": level, "gen": gen, "peak": round(best_score, 4)})
                level += 1
                task = LEVELS[level]
                pop = [grow_brain(best, task["layers"])] + \
                      [make_brain(None, task["layers"]) for _ in range(pop_size - 1)]
                streak = 0
                tag = f"  *** ASCENDED to level {level} layers={task['layers']} at gen {gen} ***"
        else:
            streak = 0
        if gen % 10 == 0 or gen == generations - 1 or tag:
            print(f"gen {gen:4d}  level {level}  best={best_score:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s{tag}", flush=True)

        nxt = [best]
        while len(nxt) < pop_size:
            parent = random.choice(scored[:16])[1]
            nxt.append(mutate(parent, rate=0.12, sigma=max(0.02, 0.4 / (1 + gen / 60))))
        pop = nxt

    out = {
        "final_level": level,
        "seed_level": seed.get("level") if seed and seed.get("w1") else None,
        "levels": [l["layers"] for l in LEVELS],
        "ascensions": ascensions,
        "scores": best_scores,
    }
    with open(LOGDIR / "evolve_scores.json", "w") as f:
        json.dump(out, f)
    with open(LOGDIR / "evolve_best.json", "w") as f:
        json.dump({"level": level, "layers": task["layers"], **best}, f)
    print(f"done. level {level}, peak={max(best_scores):.4f}, "
          f"ascensions={len(ascensions)}  -> {LOGDIR/'evolve_scores.json'}")
    return best_scores


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(n)
