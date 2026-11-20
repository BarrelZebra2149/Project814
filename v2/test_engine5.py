"""
Acceptance tests for engine5 / moves5 (5-digit extension).   python test_engine5.py [data_gen_dir]

E1  tmax=9999 engine5 == engine (old) on 300 grids (K, H) and on 3000 random moves (dH, K')
E2  tmax=12000: K (capped at tmax) and hole count == core814 (core.is_formable over 1..tmax), 400 grids
E3  tmax=12000: 100k committed moves vs full recomputation (cnt, cntx, H, first_hole) at 20 checkpoints
E4  tmax=12000: eval_move K' == core814 K' for all single-cell changes of 20 grids (20,160 changes)
E5  moves5.fixes_for exact for 5-digit numbers (brute force) and rejected moves restore state exactly
SPEED  steps/s at tmax 9999 vs 12000
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
import engine as old  # noqa: E402
import engine5 as e5  # noqa: E402
import moves5 as mv5  # noqa: E402

R, C, NC = 8, 14, 112


@njit(cache=True)
def holes_core(g2d, tmax):
    dm = np.zeros((10, R), dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    core.build_dmask(g2d, dm)
    h = 0
    first = tmax + 1
    for n in range(1, tmax + 1):
        nd = core.fill_digits(n, buf)
        if not core.is_formable(dm, buf, nd):
            h += 1
            if n < first:
                first = n
    return h, first


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return bool(ok)


def main(data_dir):
    rng = np.random.default_rng(77)
    ok = True
    gs = []
    for f in sorted(glob.glob(f"{data_dir}/gen_*.npz")):
        z = np.load(f); m = z["k"] >= 3000; gs.append(z["grids"][m])
    G = np.concatenate(gs) if gs else np.zeros((0, R, C), np.uint8)
    pool = [rng.integers(0, 10, size=(R, C)).astype(np.uint8) for _ in range(300)] + list(G[:300])

    print("E1: engine5(tmax=9999) == engine")
    a = old.Engine(); b = e5.Engine(tmax=9999)
    bad = 0
    for g in pool[:300]:
        a.set_grid(g); b.set_grid(g)
        bad += (a.K != b.K) or (a.H != b.H)
    badm = 0
    g = pool[-1]; a.set_grid(g); b.set_grid(g)
    for _ in range(3000):
        x = int(rng.integers(NC)); v = int(rng.integers(10))
        if v == a.grid[x]:
            continue
        badm += a.peek(x, v) != b.peek(x, v)
    ok &= check("same K, H on 300 grids and same (dH, K') on 3000 moves", bad == 0 and badm == 0, f"({bad}, {badm} mismatches)")

    print("E2: tmax=12000 vs core814")
    tm = 12000
    e = e5.Engine(tmax=tm)
    bad = 0
    for g in pool[:400]:
        e.set_grid(g)
        h, first = holes_core(np.ascontiguousarray(g), tm)
        bad += (e.H != h) or (e.fh != first)
    ok &= check("hole count and first hole match core814.is_formable", bad == 0, f"(400 grids, {bad} mismatches)")

    print("E3: 100k committed moves vs recomputation (tmax=12000)")
    e.set_grid(rng.integers(0, 10, size=(R, C)))
    ref = e5.Engine(tmax=tm)
    nbad = 0
    for step in range(1, 100_001):
        x = int(rng.integers(NC)); v = int(rng.integers(10))
        if v == e.grid[x]:
            continue
        dH, Kp = e.peek(x, v); H0 = e.H
        e.commit(x, v)
        assert e.H == H0 + dH and e.K == Kp
        if step % 5000 == 0:
            ref.set_grid(e.grid)
            nbad += not (np.array_equal(e.cnt, ref.cnt) and np.array_equal(e.cntx, ref.cntx)
                         and e.H == ref.H and e.fh == ref.fh)
    ok &= check("incremental == recomputed at 20 checkpoints", nbad == 0, f"({nbad} bad)")

    print("E4: eval_move vs core814 for every single-cell change (tmax=12000)")
    tot = wrong = 0
    idx = rng.choice(len(G), size=min(20, len(G)), replace=False) if len(G) else []
    for i in idx:
        g = G[i].copy(); e.set_grid(g)
        flat = g.reshape(-1).copy()
        for x in range(NC):
            for v in range(10):
                if v == flat[x]:
                    continue
                dH, Kp = e.peek(x, v)
                old_v = flat[x]; flat[x] = v
                h, first = holes_core(np.ascontiguousarray(flat.reshape(R, C)), tm)
                flat[x] = old_v
                tot += 1
                wrong += (min(first - 1, tm) != Kp)
    ok &= check("K' matches core814", wrong == 0, f"({tot} changes, {wrong} wrong)")

    print("E5: moves5 fixes_for (5-digit) exact + rejected moves restore state")
    bad = tested = 0
    for base in (pool[0], pool[1], G[0] if len(G) else pool[2]):
        s = mv5.Searcher(base, seed=1, tmax=tm, temp=1e-9)
        holes5 = [n for n in range(10000, tm + 1) if s.e.cnt[n] == 0]
        picks = [int(n) for n in rng.choice(holes5, size=min(6, len(holes5)), replace=False)] if holes5 else []
        for n in picks:
            cands = {(x, v) for x, v, _ in s.candidates(n)}
            brute = set()
            for x in range(NC):
                oldv = int(s.e.grid[x])
                for v in range(10):
                    if v == oldv:
                        continue
                    s.e.commit(x, v)
                    if s.e.cnt[n] > 0:
                        brute.add((x, v))
                    s.e.commit(x, oldv)
            tested += 1
            bad += cands != brute
    ok &= check("5-digit candidate set == brute force", bad == 0 and tested >= 12, f"({tested} five-digit numbers, {bad} bad)")
    s = mv5.Searcher(G[0] if len(G) else pool[0], seed=1, tmax=tm, temp=1e-9)
    for n in ():
        if n > tm or s.e.cnt[n] != 0:
            continue
        cands = {(x, v) for x, v, _ in s.candidates(n)}
        brute = set()
        for x in range(NC):
            oldv = int(s.e.grid[x])
            for v in range(10):
                if v == oldv:
                    continue
                s.e.commit(x, v)
                if s.e.cnt[n] > 0:
                    brute.add((x, v))
                s.e.commit(x, oldv)
        tested += 1
        bad += cands != brute
    ok &= check("candidate set == brute force for unreadable numbers", bad == 0, f"({tested} numbers, {bad} bad)")
    nrev = nbad = 0
    for t in range(15):
        s = mv5.Searcher(G[rng.integers(len(G))] if len(G) else pool[0], seed=200 + t, tmax=tm, temp=1e-9)
        for name in ("repair", "reinforce", "extend"):
            for _ in range(5):
                before = (s.e.grid.copy(), s.e.cnt.copy(), s.e.cntx.copy(), s.e.H, s.e.fh)
                if not getattr(s, name)():
                    nrev += 1
                    after = (s.e.grid, s.e.cnt, s.e.cntx, s.e.H, s.e.fh)
                    nbad += not (np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
                                 and np.array_equal(before[2], after[2]) and before[3:] == after[3:])
    ok &= check("rejected moves leave the engine bit-identical", nbad == 0, f"({nrev} rejections, {nbad} differ)")

    print("E6: weighted hole energy  (a=3, core_mult=20, tmax=12000)")
    ew = e5.Engine(tmax=tm, a=3.0, core_mult=20.0)
    ew.set_grid(G[0] if len(G) else pool[0])
    hole_w = lambda eng: float(eng.wt[1:][eng.cnt[1:] == 0].sum())
    ok &= check("H == sum of weights of unreadable numbers", abs(ew.H - hole_w(ew)) < 1e-6, f"(H={ew.H:.2f})")
    bad = 0
    for _ in range(3000):
        x = int(rng.integers(NC)); v = int(rng.integers(10))
        if v == ew.grid[x]:
            continue
        dH, Kp = ew.peek(x, v)
        H0 = ew.H
        ew.commit(x, v)
        bad += abs((ew.H - H0) - dH) > 1e-6 or abs(ew.H - hole_w(ew)) > 1e-6
    ok &= check("peek dH == actual change; H == recomputed after every commit", bad == 0, f"({bad} bad)")
    for _ in range(50_000):
        x = int(rng.integers(NC)); v = int(rng.integers(10))
        if v != ew.grid[x]:
            ew.commit(x, v)
    drift = abs(ew.H - hole_w(ew))
    ok &= check("float drift after 50k commits is negligible", drift < 1e-6, f"(drift {drift:.2e})")
    # weighted anneal chunk keeps H consistent as well
    best = np.zeros(NC, np.int8)
    ew.anneal(20000, 0.5, 3, best)
    ok &= check("jitted anneal keeps weighted H consistent", abs(ew.H - hole_w(ew)) < 1e-6, f"(diff {abs(ew.H-hole_w(ew)):.2e})")
    nrev = nbad = 0
    for t in range(10):
        sw = mv5.Searcher(G[rng.integers(len(G))] if len(G) else pool[0], seed=400 + t, tmax=tm, temp=1e-9, a=3.0, core_mult=20.0)
        for name in ("repair", "reinforce", "extend", "scan_move"):
            for _ in range(4):
                before = (sw.e.grid.copy(), sw.e.cnt.copy(), sw.e.cntx.copy(), sw.e.H, sw.e.fh)
                moved = getattr(sw, name)()
                if not moved:
                    nrev += 1
                    after = (sw.e.grid, sw.e.cnt, sw.e.cntx, sw.e.H, sw.e.fh)
                    nbad += not (np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
                                 and np.array_equal(before[2], after[2]) and abs(before[3] - after[3]) < 1e-6 and before[4] == after[4])
    ok &= check("rejected moves restore grid/cnt/cntx exactly (H within 1e-6)", nbad == 0, f"({nrev} rejections, {nbad} differ)")

    print("SPEED (random-proposal annealing, 1 thread)")
    for tmx in (9999, 12000):
        e = e5.Engine(pool[-1], tmax=tmx)
        best = np.zeros(NC, np.int8)
        e.anneal(2000, 1.0, 1, best)
        n = 500_000
        t = time.time(); e.anneal(n, 1.0, 2, best); dt = time.time() - t
        print(f"  tmax={tmx}: {n/dt:,.0f} steps/s   (old engine measured 29,000/s at tmax=9999)")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data_gen"))
