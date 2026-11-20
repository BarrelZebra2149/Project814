"""
Coverage statistics: how many 4-digit strings does a grid cover, how many walks
produce them (multiplicity), and how far are our grids from covering 1000..T?
Only our own generated data (gen_data.py output) + fresh random grids.

    python cover_stats.py data_gen
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402

ROWS, COLS = core.ROWS, core.COLS


@njit(cache=True)
def cover4(grid):
    """-> (covered[10000] bool over all 4-digit strings incl. leading zeros,
           n_walks = number of directed 4-cell king walks)."""
    dm = np.zeros((10, ROWS), dtype=np.int64)
    core.build_dmask(grid, dm)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    covered = np.zeros(10000, dtype=np.bool_)
    for s in range(10000):
        buf[0] = s // 1000; buf[1] = (s // 100) % 10; buf[2] = (s // 10) % 10; buf[3] = s % 10
        covered[s] = core.is_formable(dm, buf, 4)
    # directed 4-cell walks (3 king steps), cells may repeat
    walks = 0
    for r in range(ROWS):
        for c in range(COLS):
            # count 3-step walks from (r,c) with a tiny DP over neighbour counts
            pass
    return covered


def n_walks():
    """directed 4-cell walks in the 8x14 king graph (independent of digits)."""
    nb = np.zeros((ROWS * COLS, ROWS * COLS), dtype=np.int64)
    for r in range(ROWS):
        for c in range(COLS):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
                        nb[r * COLS + c, (r + dr) * COLS + c + dc] = 1
    return int((nb @ nb @ nb).sum())


def load_climbed(d, kmin=3000):
    gs, ks = [], []
    for f in sorted(glob.glob(f"{d}/gen_*.npz")):
        z = np.load(f); m = z["k"] >= kmin
        gs.append(z["grids"][m]); ks.append(z["k"][m])
    return np.concatenate(gs), np.concatenate(ks)


def summarize(name, grids, T=7666):
    unc_to_T, unc_1k, distinct, lead = [], [], [], np.zeros(10)
    for g in grids:
        cov = cover4(np.ascontiguousarray(g))
        need = np.arange(1000, T + 1)
        miss = need[~cov[need]]
        # a number is also fine if its reversal is covered (walks are reversible)
        unc_to_T.append(len(miss))
        unc_1k.append(int((~cov[1000:]).sum()))
        distinct.append(int(cov.sum()))
        for m in miss[:50]:
            lead[m // 1000] += 1
    print(f"{name}: n={len(grids)}  distinct 4-strings covered: mean {np.mean(distinct):.0f}/10000"
          f"  | uncovered in [1000,{T}]: mean {np.mean(unc_to_T):.0f}, min {np.min(unc_to_T)}"
          f"  | uncovered in [1000,9999]: mean {np.mean(unc_1k):.0f}")
    return np.mean(distinct)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data_gen"
    W = n_walks()
    print(f"directed 4-cell walks in 8x14 king graph: {W}  (undirected {W // 2})")
    rng = np.random.default_rng(5)
    rnd = rng.integers(0, 10, size=(300, ROWS, COLS)).astype(np.uint8)
    dr = summarize("random grids   ", rnd)
    G, K = load_climbed(d)
    idx = rng.choice(len(G), size=min(300, len(G)), replace=False)
    dc = summarize(f"climbed K>=3000", G[idx])
    print(f"\nmultiplicity (walks per distinct covered string): random {W / dr:.2f}, climbed {W / dc:.2f}")
    print(f"ideal for K=7666: every one of the 6667 strings in [1000,7666] covered "
          f"(~{6667 / (W / 2) * 100:.0f}% of undirected walks must hit a distinct needed string)")
    lam = W / 10000
    print(f"random-grid Poisson model: {lam:.2f} walks per string -> P(string uncovered) ~ {np.exp(-lam):.3f} "
          f"-> expected uncovered in [1000,7666] ~ {6667 * np.exp(-lam):.0f}")
