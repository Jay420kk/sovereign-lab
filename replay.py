#!/usr/bin/env python3
"""replay.py — render an SVG "hunting replay" of the best island brain.

Loads logs/evolve_best.json (the best brain committed by the latest island),
simulates ONE trial of its task (static homing or moving-target pursuit) with
a fixed seed, and renders an SVG animation of agent vs food over time.

The result is embedded in the status page so you can actually WATCH what the
evolved brain does instead of trusting a score number.

Output: logs/replay.svg
"""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
sys.path.insert(0, str(ROOT / "evolve"))

import evolve_mind as em  # noqa: E402

FRAMES_CAP = 400  # cap step count for a fast page load (subsample long runs)


def render_trial(brain, task, seed=7):
    """Simulate one trial; return list of (t, ax, ay, fx, fy, px, py)."""
    rng = random.Random(seed)
    task_type = task["task"]
    moving = task_type == "moving"
    hunted = task_type == "hunted"
    speed = task.get("speed", 0.0)
    pspeed = task.get("predator_speed", 0.0)
    catch_r = task.get("catch_r", 1.5)
    fx = rng.uniform(-task["food_r"], task["food_r"])
    fy = rng.uniform(-task["food_r"], task["food_r"])
    vx = vy = 0.0
    if moving and speed > 0:
        ang = rng.uniform(0, 2 * math.pi)
        vx, vy = math.cos(ang) * speed, math.sin(ang) * speed
    px = py = pvx = pvy = 0.0
    if hunted:
        ang = rng.uniform(0, 2 * math.pi)
        px, py = math.cos(ang) * task["clamp"] * 0.9, math.sin(ang) * task["clamp"] * 0.9
    ax, ay = 0.0, 0.0
    pts = [(0.0, ax, ay, fx, fy, px, py)]
    caught = False
    for s in range(task["steps"]):
        if hunted:
            dxp, dyp = ax - px, ay - py
            pn = math.hypot(dxp, dyp) or 1.0
            pvx, pvy = dxp / pn * pspeed, dyp / pn * pspeed
            px += pvx
            py += pvy
            if math.hypot(ax - px, ay - py) < catch_r:
                caught = True
        sens = [ax / task["clamp"], ay / task["clamp"],
                fx / task["clamp"], fy / task["clamp"]]
        if moving:
            sens += [vx / speed if speed else 0.0, vy / speed if speed else 0.0]
        if hunted:
            sens += [px / task["clamp"], py / task["clamp"],
                     pvx / pspeed if pspeed else 0.0, pvy / pspeed if pspeed else 0.0]
        sens += [math.sin(0.05 * s), math.cos(0.05 * s)]
        dx, dy = em.brain_fwd(brain, sens)
        ax += 0.25 * dx
        ay += 0.25 * dy
        ax = max(-task["clamp"], min(task["clamp"], ax))
        ay = max(-task["clamp"], min(task["clamp"], ay))
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
        pts.append((s + 1, ax, ay, fx, fy, px, py))
        if caught:
            break
    return pts, caught


def build_svg(brain, task):
    pts, caught = render_trial(brain, task)
    if len(pts) > FRAMES_CAP:
        step = len(pts) // FRAMES_CAP
        pts = pts[::step] + [pts[-1]]
    hunted = task["task"] == "hunted"

    clamp = task["clamp"]
    pad = 18
    vw, vh = 500, 500
    scale = (vw - 2 * pad) / (2 * clamp)

    def X(v):
        return pad + (v + clamp) * scale

    def Y(v):
        return vh - pad - (v + clamp) * scale

    # trails (subsample to ~90 visible points for legibility)
    trail_n = max(2, min(90, len(pts) // 4))
    atrail = " ".join("{:.1f},{:.1f}".format(X(ax), Y(ay)) for _, ax, ay, _, _, _, _ in pts[::trail_n])
    ftrail = " ".join("{:.1f},{:.1f}".format(X(fx), Y(fy)) for _, _, _, fx, fy, _, _ in pts[::trail_n])
    ptrail = ""
    if hunted:
        ptrail = " ".join("{:.1f},{:.1f}".format(X(px), Y(py)) for _, _, _, _, _, px, py in pts[::trail_n])

    final_ax, final_ay = pts[-1][1], pts[-1][2]
    final_fx, final_fy = pts[-1][3], pts[-1][4]
    d = math.hypot(final_ax - final_fx, final_ay - final_fy)

    # score of this exact replay trial
    replay_score = 1.0 / (1.0 + d / task["dist_scale"])
    if hunted and caught:
        replay_score *= 0.3

    lines = []
    lines.append("<svg viewBox='0 0 {0} {1}' width='100%'>".format(vw, vh))
    # arena boundary
    lines.append("<rect x='{:.1f}' y='{:.1f}' width='{:.1f}' height='{:.1f}' "
                 "fill='#141a20' stroke='#2a3540'/>".format(
                     pad, pad, vw - 2 * pad, vh - 2 * pad))
    status = "CAUGHT" if (hunted and caught) else ("replay score={:.3f}".format(replay_score))
    lines.append("<text x='{x:.0f}' y='14' fill='#7c8698' font-size='11' "
                 "font-family='monospace'>agent (blue) vs food (rose){} &middot; "
                 "task={task} &middot; {status}</text>".format(
                     " vs predator (red)" if hunted else "",
                     x=vw / 2, task=task["task"], status=status))
    # food trail + food marker
    lines.append("<polyline points='{0}' fill='none' stroke='#b48ead' "
                 "stroke-width='1.5' opacity='0.55'/>".format(ftrail))
    lines.append("<circle cx='{:.1f}' cy='{:.1f}' r='7' fill='none' stroke='#b48ead' "
                 "stroke-width='2'/>".format(X(final_fx), Y(final_fy)))
    lines.append("<text x='{:.1f}' y='{:.1f}' fill='#b48ead' font-size='9' "
                 "font-family='monospace'>food</text>".format(
                     X(final_fx) + 10, Y(final_fy) - 8))
    # predator trail + marker
    if hunted:
        final_px, final_py = pts[-1][5], pts[-1][6]
        lines.append("<polyline points='{0}' fill='none' stroke='#e06c75' "
                     "stroke-width='1.5' opacity='0.5'/>".format(ptrail))
        lines.append("<path d='M{0} L{1}' stroke='#e06c75' stroke-width='2'/>".format(
            "{:.0f} {:.0f}".format(X(final_px), Y(final_py)),
            "{:.0f} {:.0f}".format(X(final_ax), Y(final_ay))))
        lines.append("<circle cx='{:.1f}' cy='{:.1f}' r='6' fill='none' stroke='#e06c75' "
                     "stroke-width='2'/>".format(X(final_px), Y(final_py)))
        lines.append("<text x='{:.1f}' y='{:.1f}' fill='#e06c75' font-size='9' "
                     "font-family='monospace'>predator</text>".format(
                         X(final_px) + 10, Y(final_py) - 10))
    # agent trail + agent marker
    lines.append("<polyline points='{0}' fill='none' stroke='#7fb3d5' "
                 "stroke-width='2'/>".format(atrail))
    lines.append("<circle cx='{:.1f}' cy='{:.1f}' r='5' fill='#7fb3d5'/>".format(
        X(final_ax), Y(final_ay)))
    lines.append("<text x='{:.1f}' y='{:.1f}' fill='#7fb3d5' font-size='9' "
                 "font-family='monospace'>agent</text>".format(
                     X(final_ax) + 10, Y(final_ay) + 14))
    lines.append("</svg>")
    return "\n".join(lines)


def main():
    best = json.loads((LOGS / "evolve_best.json").read_text())
    level = max(0, int(best.get("level") or 0))
    task = em.level_config(level)
    if "w1" not in best or "w2" not in best:
        print("[replay] no usable brain — skipping")
        return
    svg = build_svg(best, task)
    LOGS.mkdir(exist_ok=True)
    (LOGS / "replay.svg").write_text(svg)
    print("[replay] logs/replay.svg written (level {}, task {})".format(
        level, task["task"]))


if __name__ == "__main__":
    main()
