#!/usr/bin/env python3
"""worker.py — time-bounded lab worker for GitHub Actions.

Runs one complete worker "island" within a strict time budget (the
Actions job is capped, so we must always finish and commit results).

  1. evolve_mind  — evolve a fresh 6-6-2 neural population
  2. math_discovery — brute-force integer relations among constants
  3. writes JSON results to logs/ for the workflow to commit

Pure Python stdlib — no numpy, no model, no secrets. Safe to run on
any Linux runner with Python 3.

Usage: python3 worker.py [time_budget_minutes]
"""

import json
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


def main():
    budget_min = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    deadline = time.time() + budget_min * 60
    meta = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget_min": budget_min,
        "platform": sys.platform,
    }
    print(f"[worker] start {meta['ts']} budget={budget_min}min", flush=True)

    # 1) evolve — pick generation count so we finish inside the budget.
    #    Home runs ~1000 gens in ~7.5h on 4 threads (~30s/gen, 50-brain
    #    population, serial scoring). The 4-core ARM runner is comparable,
    #    so budget with a safe 30s/gen estimate.
    gens = max(200, int((budget_min * 60 * 0.85) / 30))
    print(f"[worker] evolve: {gens} gens", flush=True)
    scores = evolve_mind.run(gens)
    elapsed = time.time() - deadline + budget_min * 60
    print(f"[worker] evolve done in {elapsed:.0f}s, peak={max(scores):.4f}", flush=True)
    with open(LOGDIR / "evolve_scores.json", "w") as f:
        json.dump({"meta": meta, "gens": gens, "peak": max(scores), "scores": scores}, f)
    with open(LOGDIR / "evolve_best.json", "w") as f:
        f.write((ROOT / "evolve" / "logs" / "evolve_best.json").read_text())

    # 2) math discovery — 80 digits, leave headroom for the commit step
    remaining = deadline - time.time()
    print(f"[worker] math: remaining {remaining:.0f}s", flush=True)
    math_discovery.main(80)
    with open(LOGDIR / "math_findings.json", "w") as f:
        f.write((ROOT / "math-discovery" / "logs" / "math_findings.json").read_text())

    print("[worker] done", flush=True)


if __name__ == "__main__":
    main()
