"""
Benchmarks for the sa814 rewrite:

  1. Scorer throughput: old DFS-based _evaluate_core_numba (from
     code/814_cpu_count_first.py) vs new bitmask core814.evaluate, evals/sec
     on a fixed set of real corpus grids.
  2. SA iteration throughput: core814.run_block iters/sec at 1 thread vs
     however many threads the machine has (scaling check).

Usage:  python bench.py [--grids N] [--sa-seconds S]
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numba
import numpy as np

import core814 as core
import seeding

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = REPO_ROOT / "data"


def load_reference():
    spec = importlib.util.spec_from_file_location("orig_count_first", CODE_DIR / "814_cpu_count_first.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench_scorer(n_grids: int):
    print(f"\n=== Scorer throughput: old DFS vs new bitmask ({n_grids} real grids) ===")
    orig = load_reference()
    grids = seeding.load_corpus(DATA_DIR, None, None)[:n_grids]
    if not grids:
        print("  no corpus grids found under data/*.txt, skipping.")
        return
    print(f"  using {len(grids)} grids from data/*.txt")

    # warm up JIT for both
    g0 = grids[0].astype(np.int64).reshape(-1)
    orig._evaluate_core_numba(g0, core.ROWS, core.COLS)
    dmask0 = np.zeros((10, core.ROWS), dtype=np.int64)
    core.build_dmask(grids[0], dmask0)
    stamp0 = np.full(core.UPPER, -1, dtype=np.int64)
    buf0 = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    core.evaluate(dmask0, stamp0, 1, 400, 1000, 10000, True, buf0)

    t0 = time.perf_counter()
    for g in grids:
        flat = g.astype(np.int64).reshape(-1)
        orig._evaluate_core_numba(flat, core.ROWS, core.COLS)
    t_old = time.perf_counter() - t0

    dmask = np.zeros((10, core.ROWS), dtype=np.int64)
    stamp = np.full(core.UPPER, -1, dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    t0 = time.perf_counter()
    gen = 1
    for g in grids:
        core.build_dmask(g, dmask)
        gen += 1
        core.evaluate(dmask, stamp, gen, 400, 1000, 10000, True, buf)
    t_new = time.perf_counter() - t0

    print(f"  old (DFS, count_first):  {len(grids)/t_old:9.1f} evals/sec  ({t_old:.3f}s total)")
    print(f"  new (bitmask, core814):  {len(grids)/t_new:9.1f} evals/sec  ({t_new:.3f}s total)")
    print(f"  speedup: {t_old/t_new:.2f}x")


def _make_replica_state(R, seed_grids):
    dmasks = np.zeros((R, 10, core.ROWS), dtype=np.int64)
    grids = np.stack([seed_grids[i % len(seed_grids)].copy() for i in range(R)])
    for i in range(R):
        core.build_dmask(grids[i], dmasks[i])
    stamps = np.full((R, core.UPPER), -1, dtype=np.int64)
    gens = np.ones(R, dtype=np.int64)
    digit_bufs = np.zeros((R, core.DIGIT_BUF_LEN), dtype=np.int64)
    rng_states = np.stack([core.make_rng_state(1000 + i) for i in range(R)])
    scores_arr = np.zeros(R, dtype=np.int64)
    looks_arr = np.zeros(R, dtype=np.int64)
    counts_arr = np.zeros(R, dtype=np.int64)
    energies = np.zeros(R, dtype=np.float64)
    buf0 = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    for i in range(R):
        s, l, c = core.evaluate(dmasks[i], stamps[i], 1, 400, 1000, 10000, False, buf0)
        scores_arr[i], looks_arr[i], counts_arr[i] = s, l, c
        energies[i] = core.energy_of(s, l, c, 0.0, 0.0, 1.0, 0.002, 0.0, 0.0, 0.0)
    temps = np.full(R, 1.0)
    lahc_len = 32
    hist = np.zeros((R, lahc_len), dtype=np.float64)
    hist[:] = energies[:, None]
    lahc_pos = np.zeros(R, dtype=np.int64)
    accept_counter = np.zeros(R, dtype=np.int64)
    move_counter = np.zeros((R, core.N_MOVES), dtype=np.int64)
    swap_accept = np.zeros(max(R - 1, 0), dtype=np.int64)
    swap_attempt = np.zeros(max(R - 1, 0), dtype=np.int64)
    return dict(grids=grids, dmasks=dmasks, stamps=stamps, gens=gens, digit_bufs=digit_bufs,
                rng_states=rng_states, scores_arr=scores_arr, looks_arr=looks_arr,
                counts_arr=counts_arr, energies=energies, temps=temps, hist=hist,
                lahc_pos=lahc_pos, accept_counter=accept_counter, move_counter=move_counter,
                swap_accept=swap_accept, swap_attempt=swap_attempt)


def bench_sa_scaling(seconds: float):
    print(f"\n=== SA iteration throughput scaling ({seconds:.1f}s per measurement) ===")
    seed_grids = seeding.load_corpus(DATA_DIR, None, None)[:8]
    if not seed_grids:
        seed_grids = [np.random.default_rng(0).integers(0, 10, (8, 14)).astype(np.uint8)]

    edge_pos = core.build_edge_positions()
    move_probs = np.array([0.34, 0.20, 0.20, 0.20, 0.01, 0.05])
    swap_rng = core.make_rng_state(1)

    max_threads = numba.config.NUMBA_NUM_THREADS
    for threads in sorted(set([1, max(1, max_threads // 2), max_threads])):
        numba.set_num_threads(threads)
        R = threads
        st = _make_replica_state(R, seed_grids)
        # warm-up / JIT compile
        core.run_block(st["grids"], st["dmasks"], st["stamps"], st["gens"], st["energies"],
                        st["scores_arr"], st["looks_arr"], st["counts_arr"], st["temps"], st["rng_states"],
                        st["digit_bufs"],
                        st["hist"], st["lahc_pos"], swap_rng,
                        edge_pos, move_probs, 0.5,
                        1.0, 0.002, 0.0, 0.0, 0.0, False, 400, 1000, 10000,
                        core.ACCEPT_SA, 500, 2, True,
                        st["accept_counter"], st["move_counter"], st["swap_accept"], st["swap_attempt"])

        st["accept_counter"][:] = 0
        st["move_counter"][:] = 0
        t0 = time.perf_counter()
        done = 0
        iters_per_segment = 2000
        while time.perf_counter() - t0 < seconds:
            core.run_block(st["grids"], st["dmasks"], st["stamps"], st["gens"], st["energies"],
                            st["scores_arr"], st["looks_arr"], st["counts_arr"], st["temps"], st["rng_states"],
                            st["digit_bufs"],
                            st["hist"], st["lahc_pos"], swap_rng,
                            edge_pos, move_probs, 0.5,
                            1.0, 0.002, 0.0, 0.0, 0.0, False, 400, 1000, 10000,
                            core.ACCEPT_SA, iters_per_segment, 5, True,
                            st["accept_counter"], st["move_counter"], st["swap_accept"], st["swap_attempt"])
            done += R * iters_per_segment * 5
        elapsed = time.perf_counter() - t0
        print(f"  threads={threads:>2d} replicas={R:>2d}: {done/elapsed:10.0f} iters/sec total "
              f"({done/elapsed/R:8.0f} iters/sec/replica)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, default=150)
    ap.add_argument("--sa-seconds", type=float, default=3.0)
    args = ap.parse_args()

    bench_scorer(args.grids)
    bench_sa_scaling(args.sa_seconds)


if __name__ == "__main__":
    main()
