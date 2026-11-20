"""
Is the HOLE COUNT a smoother landscape than K?  (the PLAN's energy rests on this)
For every single-cell change, compare  dK  vs  d(holes in [1,T]).

  python hole_smooth.py data_gen        # our own K~3100 grids, T = K+1500
  python hole_smooth.py corpus ../data  # READ-ONLY high-K measurement (user-approved)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402
import fragility as fr  # noqa: E402

R, C = core.ROWS, core.COLS


@njit(cache=True)
def count_holes(dm, buf, T):
    h = 0
    for n in range(1, T + 1):
        nd = core.fill_digits(n, buf)
        if not core.is_formable(dm, buf, nd):
            h += 1
    return h


@njit(cache=True)
def single_changes(grid, T, out_dh):
    dm = np.zeros((10, R), dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    core.build_dmask(grid, dm)
    h0 = count_holes(dm, buf, T)
    i = 0
    for r in range(R):
        for c in range(C):
            o = grid[r, c]
            for v in range(10):
                if v == o:
                    continue
                core.set_cell(grid, dm, r, c, v)
                out_dh[i] = count_holes(dm, buf, T) - h0
                i += 1
                core.set_cell(grid, dm, r, c, o)
    return h0


def report(name, grids, sc, add):
    dh_all, dk_all, h0s = [], [], []
    for g in grids:
        g = np.ascontiguousarray(g.copy())
        K = sc(g); T = min(9999, K + add)
        out = np.zeros(R * C * 9, dtype=np.int64)
        h0s.append(single_changes(g, T, out))
        dh_all.append(out)
        dk = []
        flat = g.reshape(-1)
        for x in range(R * C):
            old = flat[x]
            for v in range(10):
                if v == old:
                    continue
                flat[x] = v; dk.append(sc(g) - K); flat[x] = old
        dk_all.append(np.array(dk))
    dh = np.concatenate(dh_all); dk = np.concatenate(dk_all)
    p = lambda a, q: int(np.percentile(a, q))
    print(f"\n{name}: {len(grids)} grids, {len(dh)} single-cell changes, baseline holes in [1,K+{add}]: median {int(np.median(h0s))}")
    print(f"  dK       : median {p(dk,50):6d}  10% {p(dk,10):6d}  1% {p(dk,1):6d}   P(dK<-300)={np.mean(dk<-300):.1%}")
    print(f"  d(holes) : median {p(dh,50):6d}  90% {p(dh,90):6d}  99% {p(dh,99):6d}  max {int(dh.max())}")
    for lim in (0, 2, 5, 10, 20, 50):
        print(f"    P(d(holes) <= {lim:2d}) = {np.mean(dh <= lim):.1%}")
    print(f"  corr(dK, d(holes)) = {np.corrcoef(dk, dh)[0,1]:.2f}")


if __name__ == "__main__":
    sc = fr.Scorer()
    rng = np.random.default_rng(2)
    if sys.argv[1] == "corpus":
        import fragility_hi as fh
        gs = fh.read_grids(sys.argv[2])
        ks = np.array([sc(g) for g in gs])
        for lo, hi, label in ((6000, 7000, "corpus K in [6000,7000)"), (7000, 8000, "corpus K >= 7000")):
            idx = np.where((ks >= lo) & (ks < hi))[0]
            if len(idx):
                pick = rng.choice(idx, size=min(8, len(idx)), replace=False)
                report(label, [gs[i] for i in pick], sc, 2000)
    else:
        gs, kk = [], []
        for f in sorted(glob.glob(f"{sys.argv[1]}/gen_*.npz")):
            z = np.load(f); m = z["k"] >= 3000; gs.append(z["grids"][m])
        G = np.concatenate(gs)
        pick = rng.choice(len(G), size=20, replace=False)
        report("our data K~3100", [G[i] for i in pick], sc, 1500)
