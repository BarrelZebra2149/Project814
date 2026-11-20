"""
Stats-data generator (P1, from scratch -- NO corpus, NO old SA results).

Runs neutral-move-allowed hill climbing from all-zero grids and from random
grids, snapshotting (grid, K, look) along every trajectory. Purpose: collect
our own labelled data to test whether score is predictable from grid features
(PLAN.md section 4), and to see which numbers are the bottleneck (K+1).

    python gen_data.py --seconds 600 --workers 7 --out data_gen --seed 1000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402

ROWS, COLS = core.ROWS, core.COLS
LOOK_WINDOW = 400


@njit(cache=True)
def climb(grid, steps, seed, rec_every, rec_grids, rec_k, rec_look, rec_n0):
    """Hill-climb with neutral moves. Objective = K*1000 + look (look<=400).
    Returns (number of records written, final K)."""
    np.random.seed(seed)
    dmask = np.zeros((10, ROWS), dtype=np.int64)
    stamp = np.full(core.UPPER, -1, dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    gen = 0
    core.build_dmask(grid, dmask)
    gen += 1
    k, look, _c = core.evaluate(dmask, stamp, gen, LOOK_WINDOW, 1000, 10000, False, buf)
    cur = k * 1000 + look
    n = rec_n0
    cap = rec_grids.shape[0]
    for t in range(steps):
        r = np.random.randint(0, ROWS)
        c = np.random.randint(0, COLS)
        old = grid[r, c]
        new = np.random.randint(0, 10)
        if new != old:
            core.set_cell(grid, dmask, r, c, new)
            gen += 1
            k2, look2, _c2 = core.evaluate(dmask, stamp, gen, LOOK_WINDOW, 1000, 10000, False, buf)
            val = k2 * 1000 + look2
            if val >= cur:
                cur = val
                k, look = k2, look2
            else:
                core.set_cell(grid, dmask, r, c, old)
        # NOTE: record check is outside the `if new != old` block so that a
        # no-op draw can never skip a snapshot (every run yields exactly
        # steps // rec_every records, which the analysis relies on).
        if (t + 1) % rec_every == 0 and n < cap:
            rec_grids[n] = grid
            rec_k[n] = k
            rec_look[n] = look
            n += 1
    return n, k


def worker(args):
    wid, seconds, seed, out_dir, steps, rec_every = args
    rng = np.random.default_rng(seed + wid)
    cap = 200_000
    rec_grids = np.zeros((cap, ROWS, COLS), dtype=np.uint8)
    rec_k = np.zeros(cap, dtype=np.int64)
    rec_look = np.zeros(cap, dtype=np.int64)
    rec_start = np.zeros(cap, dtype=np.int8)   # 0 = zero start, 1 = random start
    rec_run = np.zeros(cap, dtype=np.int32)    # trajectory id within this worker (for group splits)
    n = 0
    t_end = time.time() + seconds
    runs = 0
    while time.time() < t_end and n < cap - steps // rec_every - 1:
        kind = runs % 2
        if kind == 0:
            grid = np.zeros((ROWS, COLS), dtype=np.uint8)
        else:
            grid = rng.integers(0, 10, size=(ROWS, COLS)).astype(np.uint8)
        n0 = n
        n, _k = climb(grid, steps, int(rng.integers(1, 2**31 - 1)), rec_every,
                      rec_grids, rec_k, rec_look, n)
        rec_start[n0:n] = kind
        rec_run[n0:n] = runs
        runs += 1
    # global run id = wid * 100000 + local run id
    np.savez_compressed(Path(out_dir) / f"gen_{wid}.npz", grids=rec_grids[:n], k=rec_k[:n],
                         look=rec_look[:n], start=rec_start[:n],
                         run=(wid * 100000 + rec_run[:n]).astype(np.int64))
    return wid, n, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="data_gen")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=300_000, help="steps per hill-climb run")
    ap.add_argument("--rec-every", type=int, default=1000)
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    jobs = [(i, a.seconds, a.seed, a.out, a.steps, a.rec_every) for i in range(a.workers)]
    with mp.Pool(a.workers) as p:
        for wid, n, runs in p.imap_unordered(worker, jobs):
            print(f"worker {wid}: {n} records, {runs} runs", flush=True)


if __name__ == "__main__":
    main()
