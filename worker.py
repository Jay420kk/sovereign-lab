#!/usr/bin/env python3
"""worker.py — time-bounded lab worker for GitHub Actions.

Runs one complete worker "island" within a strict time budget (the
Actions job is capped, so we must always finish and commit results).

  1. evolve_mind  — evolve a 6-6-2 neural population with open-ended
                     escalation (brain grows when the task is solved)
  2. math_discovery — brute-force integer relations among constants
  3. writes JSON results to logs/ for the workflow to commit

Pure Python stdlib — no numpy, no model, no secrets. Safe to run on
any Linux runner with Python 3.

Usage: python3 worker.py [time_budget_minutes]
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "evolve"))
sys.path.insert(0, str(ROOT / "math-discovery"))

import evolve_mind  # noqa: E402
import math_discovery  # noqa: E402


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_seed():
    """The previous island's best brain (committed at logs/evolve_best.json).

    Each island gets a fresh checkout of latest main, so this is the
    cumulative best found so far — cross-island inheritance instead of
    restarting evolution from scratch every run."""
    try:
        d = json.loads((ROOT / "logs" / "evolve_best.json").read_text())
        if isinstance(d, dict) and d.get("w1") and d.get("w2"):
            return d
    except Exception:
        pass
    return None


def planned_gens(budget_min, ncores):
    """Over-requested generation cap; the deadline is the real guard.

    evolve_mind.run() stops at whichever comes first — generations exhausted
    or the deadline. Under-requesting is the failure mode: a stale per-gen
    estimate (any fixed one drifts as levels/hardware change) lets the loop
    finish early and the unused evolve budget silently goes to math instead
    of more generations. So we ask for far more than could possibly fit and
    let the deadline bind: the multiprocessing pool's speedup then converts
    directly into more generations within the budget (the ~4x goal). The
    floor of 50 keeps tiny budgets productive. The runtime decides the real
    count completed.
    """
    workers = min(max(ncores, 1), 4)
    # optimistic floor: no generation ever costs less than ~0.5s (pop 64,
    # >=8 trials, >=300 steps — measured parallel gen is ~1.25s at L0 pop 64
    # on 4 cores, so this floor leaves ~2.5x headroom), so the cap is always
    # far above what the deadline allows -> the deadline, not this cap, is
    # the binding constraint
    optimistic = (budget_min * 60 * 0.8) / 0.5
    return max(50, int(optimistic) * workers)


def main():
    budget_min = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    deadline = time.time() + budget_min * 60
    meta = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget_min": budget_min,
        "platform": sys.platform,
    }
    print(f"[worker] start {meta['ts']} budget={budget_min}min", flush=True)

    # 1) evolve — over-requested generation cap (see planned_gens): the
    #    deadline is the real guard, so we ask for far more than fits and let
    #    it bind. evolve gets its own sub-deadline (80% of budget) so math
    #    discovery always keeps its slice; the pool's speedup then shows up as
    #    more generations actually completed within that window.
    seed = load_seed()
    ncores = max(1, os.cpu_count() or 1)
    gens = planned_gens(budget_min, ncores)
    seed_lvl = seed.get("level") if seed and "level" in seed else None
    evolve_deadline = time.time() + budget_min * 60 * 0.8
    print(f"[worker] evolve: cap {gens} gens (deadline in "
          f"{evolve_deadline - time.time():.0f}s, "
          f"seed={'L' + str(seed_lvl) if seed_lvl is not None else 'fresh'})", flush=True)
    scores = evolve_mind.run(gens, deadline=evolve_deadline, seed=seed)
    # time spent so far (== evolve time: evolve is the first stage)
    elapsed = budget_min * 60 - (deadline - time.time())
    print(f"[worker] evolve done: {len(scores)} gens in {elapsed:.0f}s, "
          f"peak={max(scores):.4f}", flush=True)
    ev = read_json(ROOT / "evolve" / "logs" / "evolve_scores.json", {}) or {}
    with open(LOGDIR / "evolve_scores.json", "w") as f:
        json.dump({
            "meta": meta, "gens": len(scores), "peak": max(scores), "scores": scores,
            "final_level": ev.get("final_level", 0),
            "seed_level": ev.get("seed_level"),
            "ascensions": ev.get("ascensions", []),
        }, f)
    with open(LOGDIR / "evolve_best.json", "w") as f:
        f.write((ROOT / "evolve" / "logs" / "evolve_best.json").read_text())

    # 2) math discovery — bounded by remaining budget
    remaining = deadline - time.time()
    print(f"[worker] math: remaining {remaining:.0f}s", flush=True)
    math_discovery.main(80, deadline=deadline)
    with open(LOGDIR / "math_findings.json", "w") as f:
        f.write((ROOT / "math-discovery" / "logs" / "math_findings.json").read_text())

    # 3) update island manifest (gen_page reads this for history)
    manifest = read_json(LOGDIR / "island_manifest.json", []) or []
    mathf = read_json(LOGDIR / "math_findings.json", {}) or {}
    manifest.append({
        "ts": meta["ts"],
        "peak": max(scores),
        "gens": len(scores),  # actual gens completed (deadline-bound), not the cap
        "level": ev.get("final_level", 0),
        "seed_level": ev.get("seed_level"),
        "ascensions": len(ev.get("ascensions", [])),
        "math_hits": len(mathf.get("hits", [])),
    })
    with open(LOGDIR / "island_manifest.json", "w") as f:
        json.dump(manifest[-50:], f, indent=1)  # keep last 50

    print("[worker] done", flush=True)


if __name__ == "__main__":
    main()
