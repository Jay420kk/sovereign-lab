#!/usr/bin/env python3
"""evolve_mind.py — open-ended evolutionary mind.

No backprop, no gradients, no human-designed task priors beyond the
physics of the arena. Just: population -> mutate -> score -> select.

The task: a 2D agent with a small network must REACH food.
Scoring is 1/(1 + dist/dist_scale) — smooth selection pressure, max 1.0.

UNBOUNDED ESCALATION: levels are generated procedurally from the level
number — there is no maximum level, so escalation never caps. Three task
types (a task switch is a genuinely new competence):

  - "static": home to a stationary food source (pure homing)
  - "moving": intercept food that moves & bounces off walls (pursuit)
    the brain gains 2 velocity sensors (8 inputs), so it must predict
    where the target WILL be, not chase where it IS.
  - "hunted": a predator chases the agent while it must still reach food
    (approach AND evasion). Brain gains predator position + velocity
    sensors (10 inputs). Predator speed rises toward the agent's max,
    so interception difficulty is unbounded even within this task.

Ascension triggers (automatic, always progresses):
  1. THRESHOLD  — best >= 0.98 sustained N gens (fast levels)
  2. PLATEAU    — no real improvement for `plateau_window` gens while
                  best >= floor (the population has saturated: push on)
  3. HARD STALL — stalled 3x window regardless of floor (never dead-end:
                  give it a bigger brain + a harder task anyway)

Sensors are arena-normalized, so scores stay comparable across levels.
The old brain's weights are embedded top-left into the bigger net, so
learned structure survives each ascent.

Usage: python3 evolve_mind.py [generations]
"""

import json
import math
import random
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent
LOGDIR = OUT_DIR / "logs"


def level_config(level):
    """Procedural level definition — UNBOUNDED. Difficulty scales with level.

    - brain: hidden width grows; input width is 6 (static) or 8 (moving)
    - arena: food_r grows geometrically; clamp and dist_scale scale with it
      (normalized resolution stays comparable, so scores remain meaningful)
    - compute caps keep per-generation cost sane at extreme levels
    - moving: target speed rises toward the agent's max speed (0.25/step),
      so required interception precision grows without bound
    """
    l = max(0, int(level))
    if l < 5:
        task_type = "static"
    elif l < 9:
        task_type = "moving"
    else:
        task_type = "hunted"
    hidden = min(6 + 2 * l, 40)
    in_w = 6 if task_type == "static" else (8 if task_type == "moving" else 10)
    food_r = 4.0 * (1.35 ** l)
    clamp = 1.5 * food_r
    dist_scale = max(1.5, food_r * (0.25 if task_type != "static" else 0.375))
    steps = min(300 + 60 * l, 1500)
    trials = min(8 + l, 20)
    speed = 0.0 if task_type == "static" else min(0.12 * (1.08 ** (l - 4)), 0.24)
    # hunted: predator speed rises toward the agent's max (0.25/step) so the
    # evasion problem grows harder without bound
    predator_speed = (min(0.14 + 0.008 * (l - 8), 0.23)
                      if task_type == "hunted" else 0.0)
    catch_r = 1.5
    return dict(task=task_type, layers=[in_w, hidden, 2],
                food_r=round(food_r, 4), steps=steps, trials=trials,
                clamp=round(clamp, 4), dist_scale=round(dist_scale, 4),
                speed=round(speed, 4), predator_speed=round(predator_speed, 4),
                catch_r=catch_r)


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
    task_type = task["task"]
    moving = task_type == "moving"
    hunted = task_type == "hunted"
    speed = task.get("speed", 0.0)
    pspeed = task.get("predator_speed", 0.0)
    catch_r = task.get("catch_r", 1.5)
    for _ in range(task["trials"]):
        fx = rng.uniform(-task["food_r"], task["food_r"])
        fy = rng.uniform(-task["food_r"], task["food_r"])
        vx = vy = 0.0
        if moving and speed > 0:
            ang = rng.uniform(0, 2 * math.pi)
            vx, vy = math.cos(ang) * speed, math.sin(ang) * speed
        # predator spawns at a random point on the arena edge, moving toward
        # the agent from the start (never spawns on top of the agent)
        px = py = 0.0
        pvx = pvy = 0.0
        if hunted:
            ang = rng.uniform(0, 2 * math.pi)
            px, py = math.cos(ang) * task["clamp"] * 0.9, math.sin(ang) * task["clamp"] * 0.9
        x, y = 0.0, 0.0
        caught = False
        min_dist = math.hypot(x - fx, y - fy)
        for s in range(task["steps"]):
            if hunted:
                # predator chases the agent's current position, clamped speed
                dxp, dyp = x - px, y - py
                pn = math.hypot(dxp, dyp) or 1.0
                pvx, pvy = dxp / pn * pspeed, dyp / pn * pspeed
                px += pvx
                py += pvy
                if math.hypot(x - px, y - py) < catch_r:
                    caught = True
            sens = [x / task["clamp"], y / task["clamp"],
                    fx / task["clamp"], fy / task["clamp"]]
            if moving:
                sens += [vx / speed if speed else 0.0, vy / speed if speed else 0.0]
            if hunted:
                sens += [px / task["clamp"], py / task["clamp"],
                         pvx / pspeed if pspeed else 0.0, pvy / pspeed if pspeed else 0.0]
            sens += [math.sin(0.05 * s), math.cos(0.05 * s)]
            dx, dy = brain_fwd(b, sens)
            x += 0.25 * dx
            y += 0.25 * dy
            x = max(-task["clamp"], min(task["clamp"], x))
            y = max(-task["clamp"], min(task["clamp"], y))
            if moving:
                fx += vx
                fy += vy
                if fx > task["clamp"]:
                    fx = 2 * task["clamp"] - fx; vx = -vx
                elif fx < -task["clamp"]:
                    fx = -2 * task["clamp"] - fx; vx = -vx
                if fy > task["clamp"]:
                    fy = 2 * task["clamp"] - fy; vy = -vy
                elif fy < -task["clamp"]:
                    fy = -2 * task["clamp"] - fy; vy = -vy
            d = math.hypot(x - fx, y - fy)
            min_dist = min(min_dist, d)
            if caught:
                break  # trial over early: caught
        trial = 1.0 / (1.0 + min_dist / task["dist_scale"])
        if hunted and caught:
            trial *= 0.3  # eaten: heavy penalty, some credit for getting close
        total += trial
    return total / task["trials"]


def _score_one(args):
    """score_brain wrapper for multiprocessing: (brain, task, rng_seed)."""
    b, task, seed = args
    return score_brain(b, task, random.Random(seed))


def _score_pop(pop, task, rng, pool):
    """Score the whole population, best-first.

    With a pool: each brain is scored in a worker with its own RNG stream, so
    food placement stays uniform and independent across brains. Without a pool:
    the classic serial loop sharing one rng. Returns [(score, brain)] sorted
    descending."""
    if pool is not None:
        seeds = [rng.randrange(1 << 30) for _ in pop]
        scores = pool.map(_score_one, [(b, task, s) for b, s in zip(pop, seeds)])
        return sorted(zip(scores, pop), key=lambda p: p[0], reverse=True)
    return sorted(((score_brain(b, task, rng), b) for b in pop),
                  key=lambda p: p[0], reverse=True)


def _make_pool():
    """Multiprocessing pool for scoring (stdlib-only, no numpy needed). Returns
    None when workers can't be spawned (e.g. Windows, or a constrained runner)
    — callers then score serially. Capped at 4 workers: the lab targets are
    4-thread machines (GitHub ARM64 runner, 2-core/4-thread laptop)."""
    try:
        import multiprocessing as mp
        n = min(4, max(1, mp.cpu_count() or 1))
        return mp.Pool(processes=n)
    except Exception:
        return None


def run(generations=200, pop_size=64, ascend_threshold=0.98, ascend_streak_req=5,
        plateau_window=20, plateau_eps=0.003, plateau_floor=0.85,
        deadline=None, seed=None):
    """Evolve; ASCEND levels automatically and without bound.

    deadline: unix ts — the loop stops cleanly at the budget, so callers
    (worker islands) always get results even if escalation slowed things down.

    seed: dict {"level", "w1", "w2", ...} — the best brain from a previous
    island (cross-island inheritance). The population starts at the seed's
    level with the old weights embedded top-left, so evolution accumulates
    across islands instead of restarting from scratch every run.

    Scoring is parallelized across cores with multiprocessing (stdlib-only),
    falling back to serial if a pool can't be created.
    """
    LOGDIR.mkdir(exist_ok=True)
    rng = random.Random()
    if seed and seed.get("w1") and seed.get("w2"):
        level = max(0, int(seed.get("level") or 0))
        task = level_config(level)
        pop = [grow_brain(seed, task["layers"])] + \
              [make_brain(None, task["layers"]) for _ in range(pop_size - 1)]
        print(f"seeded population at level {level} (task={task['task']} "
              f"layers={task['layers']}) from previous island's best", flush=True)
    else:
        level = 0
        task = level_config(level)
        pop = [make_brain(i, task["layers"]) for i in range(pop_size)]

    pool = _make_pool()
    try:
        best_scores, ascensions, level, task, best = _run_generations(
            pop, level, task, rng, pool,
            generations, pop_size, ascend_threshold, ascend_streak_req,
            plateau_window, plateau_eps, plateau_floor, deadline)
    finally:
        if pool is not None:
            pool.close()
            pool.terminate()
            pool.join()

    out = {
        "final_level": level,
        "seed_level": seed.get("level") if seed and seed.get("w1") else None,
        "task": task["task"],
        "layers": task["layers"],
        "ascensions": ascensions,
        "scores": best_scores,
    }
    with open(LOGDIR / "evolve_scores.json", "w") as f:
        json.dump(out, f)
    with open(LOGDIR / "evolve_best.json", "w") as f:
        json.dump({"level": level, "layers": task["layers"], **best}, f)
    print(f"done. level {level} (task={task['task']} layers={task['layers']}), "
          f"peak={max(best_scores):.4f}, ascensions={len(ascensions)}  -> "
          f"{LOGDIR/'evolve_scores.json'}")
    return best_scores


def _run_generations(pop, level, task, rng, pool, generations, pop_size,
                     ascend_threshold, ascend_streak_req, plateau_window,
                     plateau_eps, plateau_floor, deadline):
    """The generation loop: score (parallel if pool given) -> mutate -> select.

    Returns (best_scores, ascensions, final_level, final_task, best_brain).
    """
    best_scores, ascensions = [], []
    streak = 0
    stall = 0
    level_best = 0.0
    t0 = time.time()
    best = pop[0]  # so generations=0 still writes a valid best brain

    for gen in range(generations):
        if deadline is not None and time.time() > deadline:
            print(f"budget exhausted at gen {gen}", flush=True)
            break
        try:
            scored = _score_pop(pop, task, rng, pool)
        except Exception:
            # e.g. a worker died — degrade gracefully to serial for the rest
            print("parallel scoring failed — falling back to serial", flush=True)
            pool = None
            scored = _score_pop(pop, task, rng, None)
        best_score, best = scored[0]
        best_scores.append(best_score)
        tag = ""

        # -- ascension bookkeeping -----------------------------------------
        streak = streak + 1 if best_score >= ascend_threshold else 0
        if best_score > level_best + plateau_eps:
            level_best = best_score
            stall = 0
        else:
            stall += 1

        do_ascend = (
            streak >= ascend_streak_req          # 1: solved the current task
            or (stall >= plateau_window and best_score >= plateau_floor)  # 2: saturated
            or stall >= plateau_window * 3       # 3: hard stall — never dead-end
        )
        if do_ascend:
            ascensions.append({"level": level, "gen": gen, "peak": round(best_score, 4),
                               "task": task["task"]})
            level += 1
            task = level_config(level)
            pop = [grow_brain(best, task["layers"])] + \
                  [make_brain(None, task["layers"]) for _ in range(pop_size - 1)]
            streak = 0
            stall = 0
            level_best = 0.0
            tag = (f"  *** ASCENDED to level {level} (task={task['task']} "
                   f"layers={task['layers']}) at gen {gen} ***")

        if gen % 10 == 0 or gen == generations - 1 or tag:
            print(f"gen {gen:4d}  level {level}  best={best_score:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s{tag}", flush=True)

        nxt = [best]
        while len(nxt) < pop_size:
            parent = random.choice(scored[:16])[1]
            nxt.append(mutate(parent, rate=0.12, sigma=max(0.02, 0.4 / (1 + gen / 60))))
        pop = nxt

    return best_scores, ascensions, level, task, best


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(n)
