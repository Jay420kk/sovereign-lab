# Sovereign Lab — Distributed Worker

A distributed "island" worker for the sovereign lab. Each GitHub Actions job is
an independent island: it evolves a fresh 6-6-2 neural population and hunts for
integer relations among mathematical constants, then commits its results back
to this repo. Islands are independent — results merge via git.

- **Runners**: free GitHub-hosted ARM64 (4 vCPU / 16 GB) — free on public repos
- **Deps**: none. Pure Python 3 stdlib (`math`, `random`, `fractions`, `itertools`)
- **Schedule**: every 6 hours + manual dispatch (`Actions` tab → "Run workflow")

## Layout

```
evolve/          evolve_mind.py      — the evolutionary mind (no backprop)
math-discovery/  math_discovery.py   — PSLQ-style integer-relation search
worker.py                           — time-bounded island runner
logs/                               — committed island results (one per run)
.github/workflows/worker.yml        — the job definition
```

## Results

Each committed island stores `logs/evolve_scores.json` (best-score curve +
peak), `logs/evolve_best.json` (the winning brain), and
`logs/math_findings.json` (constant bank + any integer relations found).

## Run locally

```sh
python3 worker.py 300   # 300-minute budget
```

## Laptop sync

On the lab laptop, `sovereign-lab/bin/lab-sync.sh` pulls this repo and merges
new island results into the local lab logs.
