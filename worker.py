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
    #    Home runs ~1000 gens in ~7.5h on 4 threads (~30s/gen, 64-brain
    #    population, serial scoring). The ARM runner is ~25x faster. The
    #    deadline guarantees we stop cleanly and always write results,
    #    even if escalation slows later generations.
    gens = max(50, int((budget_min * 60 * 0.8) / 30))
    seed = load_seed()
    seed_lvl = seed.get("level") if seed else None
    print(f"[worker] evolve: {gens} gens (deadline {deadline - time.time():.0f}s, "
          f"seed={'L' + str(seed_lvl) if seed_lvl is not None else 'fresh'})", flush=True)
    scores = evolve_mind.run(gens, deadline=deadline, seed=seed)
    elapsed = budget_min * 60 - (deadline - time.time())
    print(f"[worker] evolve done in {elapsed:.0f}s, peak={max(scores):.4f}", flush=True)
    ev = read_json(ROOT / "evolve" / "logs" / "evolve_scores.json", {}) or {}
    with open(LOGDIR / "evolve_scores.json", "w") as f:
        json.dump({
            "meta": meta, "gens": gens, "peak": max(scores), "scores": scores,
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
        "gens": gens,
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
