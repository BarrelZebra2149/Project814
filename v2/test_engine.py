"""
P1 acceptance tests for engine.py  (PLAN section 4.4).  Run on the server:

    python test_engine.py data_gen
T1  engine == core814 on 1000 random grids + special grids (K, and hole count via core.is_formable)
T2  100k random committed moves  ==  full recomputation (cnt, cntx, H, first_hole), checked every 5000 moves
T3  exactness of eval_move: K after EVERY single-cell change (60 grids x 1008) == core814's K;
    plus the cut-cell model (crit) vs reality, reproducing fragility.py (97.9% equal, 0 wrong)
SPEED  steps/s of the annealing loop, cost of fragility()
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402
import engine as eg  # noqa: E402

R, C, NC = eg.ROWS, eg.COLS, eg.NC


@njit(cache=True)
def holes_via_core(grid2d):
    dm = np.zeros((10, R), dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    core.build_dmask(grid2d, dm)
    h = 0
    for n in range(1, eg.NMAX):
        nd = core.fill_digits(n, buf)
        if not core.is_formable(dm, buf, nd):
            h += 1
    return h


class Core:
    def __init__(self):
        self.dm = np.zeros((10, R), dtype=np.int64)
        self.st = np.full(core.UPPER, -1, dtype=np.int64)
        self.buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
        self.gen = 0

    def K(self, g2d):
        self.gen += 1
        core.build_dmask(np.ascontiguousarray(g2d), self.dm)
        return min(int(core.evaluate(self.dm, self.st, self.gen, 400, 1000, 10000, False, self.buf)[0]), eg.NMAX - 1)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return ok


def main(data_dir):
    cr = Core()
    rng = np.random.default_rng(123)
    allok = True

    # ------------------------------------------------------------------ T1
    print("T1: engine vs core814")
    e = eg.Engine()
    bad_k = bad_h = 0
    grids = [rng.integers(0, 10, size=(R, C)).astype(np.uint8) for _ in range(1000)]
    grids += [np.zeros((R, C), np.uint8), np.ones((R, C), np.uint8), np.full((R, C), 7, np.uint8)]
    gs = []
    for f in sorted(glob.glob(f"{data_dir}/gen_*.npz")):
        z = np.load(f); m = z["k"] >= 3000; gs.append(z["grids"][m][:200])
    grids += list(np.concatenate(gs)[:300])
    for g in grids:
        e.set_grid(g)
        bad_k += e.K != cr.K(g)
        bad_h += e.H != holes_via_core(np.ascontiguousarray(g))
    allok &= check("K matches core814", bad_k == 0, f"({len(grids)} grids, {bad_k} mismatches)")
    allok &= check("hole count matches core814.is_formable", bad_h == 0, f"({bad_h} mismatches)")

    # ------------------------------------------------------------------ T2
    print("T2: 100k committed random moves vs full recomputation")
    e.set_grid(rng.integers(0, 10, size=(R, C)))
    ref = eg.Engine()
    nbad = 0
    t0 = time.time()
    for step in range(1, 100_001):
        x = int(rng.integers(NC)); v = int(rng.integers(10))
        if v == e.grid[x]:
            continue
        dH, Kp = e.peek(x, v)
        H0 = e.H
        e.commit(x, v)
        assert e.H == H0 + dH and e.K == Kp, "peek disagrees with commit"
        if step % 5000 == 0:
            ref.set_grid(e.grid)
            same = (np.array_equal(e.cnt, ref.cnt) and np.array_equal(e.cntx, ref.cntx)
                    and e.H == ref.H and e.fh == ref.fh)
            nbad += not same
    allok &= check("incremental state == recomputed (cnt, cntx, H, first_hole) at 20 checkpoints",
                    nbad == 0, f"({nbad} bad, {time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ T3
    print("T3: eval_move exactness on every single-cell change")
    gs = []
    for f in sorted(glob.glob(f"{data_dir}/gen_*.npz")):
        z = np.load(f); m = z["k"] >= 3000; gs.append(z["grids"][m])
    G = np.concatenate(gs)
    idx = rng.choice(len(G), size=60, replace=False)
    tot = wrong = 0
    eq = up = down = 0
    for i in idx:
        g = G[i].copy()
        e.set_grid(g)
        K0 = e.K
        crit, spf, free, F = e.fragility()
        flat = g.reshape(-1).copy()
        for x in range(NC):
            for v in range(10):
                if v == flat[x]:
                    continue
                dH, Kp = e.peek(x, v)
                old = flat[x]; flat[x] = v
                kr = cr.K(flat.reshape(R, C)); flat[x] = old
                tot += 1
                wrong += Kp != kr
                pred = min(int(crit[x]), K0 + 1) - 1
                if kr == pred: eq += 1
                elif kr > pred: up += 1
                else: down += 1
    allok &= check("eval_move K' == core814 K' for all changes", wrong == 0, f"({tot} changes, {wrong} wrong)")
    allok &= check("cut-cell model never over-predicts K'", down == 0,
                    f"(equal {eq/tot:.1%}, real>pred {up/tot:.1%}, real<pred {down/tot:.2%})  [fragility.py: 97.9% / 2.1% / 0%]")

    # ------------------------------------------------------------------ speed
    print("SPEED")
    for label, g0 in (("random start", rng.integers(0, 10, size=(R, C))), ("K~3100 start", G[idx[0]])):
        e.set_grid(g0)
        best = np.zeros(NC, dtype=np.int8)
        e.anneal(2000, 15.0, 1, best)                      # compile / warm
        n = 3_000_000
        t = time.time(); acc, bestK = e.anneal(n, 15.0, 2, best); dt = time.time() - t
        print(f"  {label}: {n/dt:,.0f} steps/s  accept {acc/n:.1%}  H={e.H}  K={e.K}  best K seen={bestK}")
    t = time.time()
    for _ in range(200):
        e.fragility()
    print(f"  fragility(): {(time.time()-t)/200*1000:.2f} ms per call (K={e.K})")
    t = time.time(); e.set_grid(e.grid); print(f"  full init: {(time.time()-t)*1000:.1f} ms")
    print("\nALL PASS" if allok else "\nSOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data_gen"))
