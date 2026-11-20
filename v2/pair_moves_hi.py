"""
READ-ONLY measurement (user-approved Q1): at HIGH-score grids, does any 2-cell
compound move keep (or raise) K?  If yes, "break one cell, repair with another"
(PLAN M4) has something to work with; if no, even compound moves are blind.

All C(112,2)*81 = 503,496 two-cell changes are evaluated exactly with core814.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402
import fragility_hi as fh  # noqa: E402
import fragility as fr  # noqa: E402

R, C = core.ROWS, core.COLS


@njit(cache=True)
def all_pairs(grid, K0):
    dm = np.zeros((10, R), dtype=np.int64)
    st = np.full(core.UPPER, -1, dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    gen = 0
    core.build_dmask(grid, dm)
    n_eq = 0; n_up = 0; n_near = 0; tot = 0
    best = K0
    bx1 = by1 = bx2 = by2 = -1; bv1 = bv2 = -1
    NCELL = R * C
    for a in range(NCELL):
        r1 = a // C; c1 = a % C; o1 = grid[r1, c1]
        for v1 in range(10):
            if v1 == o1:
                continue
            core.set_cell(grid, dm, r1, c1, v1)
            for b in range(a + 1, NCELL):
                r2 = b // C; c2 = b % C; o2 = grid[r2, c2]
                for v2 in range(10):
                    if v2 == o2:
                        continue
                    core.set_cell(grid, dm, r2, c2, v2)
                    gen += 1
                    k, look, _c = core.evaluate(dm, st, gen, 400, 1000, 10000, False, buf)
                    tot += 1
                    if k > K0:
                        n_up += 1
                    elif k == K0:
                        n_eq += 1
                    elif k > K0 - 30:
                        n_near += 1
                    if k > best:
                        best = k; bx1 = a; bv1 = v1; bx2 = b; bv2 = v2
                    core.set_cell(grid, dm, r2, c2, o2)
            core.set_cell(grid, dm, r1, c1, o1)
    return tot, n_up, n_eq, n_near, best, bx1, bv1, bx2, bv2


def main(d, ks):
    grids = fh.read_grids(d)
    sc = fr.Scorer()
    by_k = {}
    for g in grids:
        by_k.setdefault(sc(g), g)
    for k in ks:
        if k not in by_k:
            print(f"K={k}: not in corpus"); continue
        g = np.ascontiguousarray(by_k[k].copy())
        t = time.time()
        tot, up, eq, near, best, a, v1, b, v2 = all_pairs(g, k)
        print(f"K={k}: {tot} two-cell changes in {time.time()-t:.0f}s | "
              f"K'>K: {up} | K'==K: {eq} | K-30<K'<K: {near} | best K'={best}"
              + (f" (cells {a}->{v1}, {b}->{v2})" if up else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
