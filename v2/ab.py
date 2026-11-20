"""
P3 A/B harness (PLAN section 9).  Same wall-clock budget per run, random-grid start, one worker per run.

  python ab.py run --out ab_out --seeds 14 --seconds 900 --workers 7 [--arms B0,B1,...]
  python ab.py report ab_out

Arms
  B0   baseline: hill climb on K*1000+look with neutral moves (what gen_data.py did), core814 scoring
  B1   engine, energy = hole count H, random single-cell proposals (jitted Metropolis), cooling schedule
  B1s  engine, H, M1 rejection-free scan move only
  B3   H, mix of M1 scan 0.5 / M3 extend 0.25 / M2 reinforce 0.25
  B4   B3 + M4 repair (mix 0.4 / 0.2 / 0.2 / 0.2)
  B5   B4 with the auxiliary energy (fragility F + free cells), aux=True
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scoring"))

SCHED = {"B1": (6.0, 0.6), "B1a": (3.0, 0.3), "B1b": (10.0, 1.0)}      # (T0, T1) guesses, compared as arms
DEFAULT_SCHED = (6.0, 0.6)
# One cycle = one fast jitted random Metropolis chunk (2000 proposals) followed by these structured moves.
# Rough time shares: B3 ~ 75% random chunk, B4 ~ 50%.
CYCLE = {
    "B3": ["extend"] * 4 + ["reinforce"] * 4,
    "B4": ["extend"] * 4 + ["reinforce"] * 4 + ["repair"] * 4,
    "B5": ["extend"] * 4 + ["reinforce"] * 4 + ["repair"] * 4,
    "B1s": ["scan_move"] * 4,
}


def temp_at(frac, arm="B3"):
    t0, t1 = SCHED.get(arm, DEFAULT_SCHED)
    return t0 * (t1 / t0) ** min(max(frac, 0.0), 1.0)


def run_one(job):
    arm, seed, seconds, out_dir = job
    rng = np.random.default_rng(10_000 + seed)
    grid = rng.integers(0, 10, size=(8, 14)).astype(np.uint8)
    t0 = time.time()
    prog = []                     # (elapsed, bestK)
    last_log = 0.0
    best_K = 0
    best_grid = grid.copy()
    extra = {}
    steps = 0

    def log(bk):
        nonlocal last_log
        now = time.time() - t0
        if now - last_log >= 30:
            prog.append((round(now), int(bk)))
            last_log = now

    if arm == "B0":
        import gen_data
        rg = np.zeros((1, 8, 14), np.uint8); rk = np.zeros(1, np.int64); rl = np.zeros(1, np.int64)
        s = int(rng.integers(1, 2**31 - 1))
        while time.time() - t0 < seconds:
            _n, k = gen_data.climb(grid, 100_000, s, 10**9, rg, rk, rl, 1)
            s += 1; steps += 100_000
            if k > best_K:
                best_K = int(k); best_grid = grid.copy()
            log(best_K)
    elif arm.startswith("B1") and arm != "B1s":
        import engine as eg
        e = eg.Engine(grid)
        bg = np.zeros(112, np.int8)
        bg[:] = e.grid
        ci = 0
        while time.time() - t0 < seconds:
            acc, bk = e.anneal(500_000, temp_at((time.time() - t0) / seconds, arm), seed * 100_000 + ci, bg)
            ci += 1; steps += 500_000
            if bk > best_K:
                best_K = bk; best_grid = bg.reshape(8, 14).copy()
            log(best_K)
        extra["final_H"] = int(e.H)
    else:
        import moves as mv
        s = mv.Searcher(grid, seed=seed, temp=DEFAULT_SCHED[0], aux=(arm == "B5"))
        cyc = CYCLE[arm]
        while time.time() - t0 < seconds:
            s.temp = temp_at((time.time() - t0) / seconds, arm)
            if arm != "B1s":
                s.rand_chunk(2000); steps += 2000
            for name in cyc:
                getattr(s, name)(); steps += 1
            log(s.best_K)
        best_K = int(s.best_K); best_grid = s.best_grid.reshape(8, 14).copy()
        extra = {"stats": s.stats, "m4_depth": {str(k): v for k, v in s.m4_depth.items()}, "final_H": int(s.e.H)}
    res = dict(arm=arm, seed=seed, seconds=round(time.time() - t0), steps=steps, best_K=int(best_K),
               progress=prog, grid=["".join(str(int(v)) for v in r) for r in np.asarray(best_grid).reshape(8, 14)], **extra)
    Path(out_dir, f"{arm}_{seed:02d}.json").write_text(json.dumps(res))
    return arm, seed, int(best_K), steps


def cmd_run(a):
    Path(a.out).mkdir(exist_ok=True)
    arms = a.arms.split(",")
    done = {p.stem for p in Path(a.out).glob("*.json")}
    jobs = [(arm, sd, a.seconds, a.out) for sd in range(a.seeds) for arm in arms if f"{arm}_{sd:02d}" not in done]
    print(f"{len(jobs)} jobs x {a.seconds}s on {a.workers} workers  (skipping {len(done)} finished)", flush=True)
    with mp.Pool(a.workers) as p:
        for arm, sd, bk, steps in p.imap_unordered(run_one, jobs):
            print(f"{time.strftime('%H:%M:%S')} {arm} seed {sd:2d}: best K = {bk}  ({steps:,} steps)", flush=True)


def cmd_report(a):
    from scipy.stats import mannwhitneyu
    res = {}
    for p in sorted(Path(a.dir).glob("*.json")):
        r = json.loads(p.read_text()); res.setdefault(r["arm"], []).append(r)
    order = [x for x in ("B0", "B1", "B1a", "B1b", "B1s", "B3", "B4", "B5") if x in res]
    print(f"{'arm':4s} {'n':>3s} {'median K':>9s} {'mean K':>8s} {'max K':>6s} {'min K':>6s} {'steps/run (median)':>20s} {'p vs B0':>9s}")
    base = [r["best_K"] for r in res.get("B0", [])]
    for arm in order:
        ks = [r["best_K"] for r in res[arm]]
        p = ""
        if arm != "B0" and base:
            p = f"{mannwhitneyu(ks, base, alternative='greater').pvalue:.3f}"
        print(f"{arm:4s} {len(ks):3d} {np.median(ks):9.0f} {np.mean(ks):8.0f} {max(ks):6d} {min(ks):6d} "
              f"{np.median([r['steps'] for r in res[arm]]):20,.0f} {p:>9s}")
    print("\nprogress (median best K at 1/4, 1/2, 3/4, end of the run):")
    for arm in order:
        cols = []
        for q in (0.25, 0.5, 0.75, 1.0):
            vals = []
            for r in res[arm]:
                pr = [k for t, k in r["progress"] if t <= q * r["seconds"]]
                vals.append(pr[-1] if pr else 0)
            cols.append(int(np.median(vals)))
        print(f"  {arm:4s} {cols}")
    for arm in ("B3", "B4", "B5"):
        if arm in res and "stats" in res[arm][0]:
            agg = {}
            for r in res[arm]:
                for k, v in r["stats"].items():
                    for kk, vv in v.items():
                        agg.setdefault(k, {}).setdefault(kk, 0)
                        agg[k][kk] += vv
            print(f"\n{arm} move stats (summed over seeds):")
            for k, v in agg.items():
                if v["tried"]:
                    print(f"  {k}: tried {v['tried']:,} accepted {v['accepted']/v['tried']:.1%} improved {v['improved']/v['tried']:.1%}")
            d = {}
            for r in res[arm]:
                for k, v in r["m4_depth"].items():
                    d.setdefault(k, [0, 0]); d[k][0] += v[0]; d[k][1] += v[1]
            if d:
                print(f"  M4 depth {{used repairs: [tried, accepted with dE<=0]}}: {dict(sorted(d.items()))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="ab_out"); r.add_argument("--seeds", type=int, default=14)
    r.add_argument("--seconds", type=float, default=900); r.add_argument("--workers", type=int, default=7)
    r.add_argument("--arms", default="B0,B1,B1a,B1b,B3,B4")
    q = sub.add_parser("report"); q.add_argument("dir")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        args.dir = args.dir; cmd_report(args)
