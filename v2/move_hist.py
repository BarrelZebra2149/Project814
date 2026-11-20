"""All 1008 single-cell changes of each base grid, bucketed by size of K change.
Tests the claim: 'most changes are killers, but some move the score a little'."""
import glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core
R, C = core.ROWS, core.COLS
dm = np.zeros((10, R), dtype=np.int64); st = np.full(core.UPPER, -1, dtype=np.int64)
buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64); gen = [0]
def score(g):
    gen[0] += 1; core.build_dmask(np.ascontiguousarray(g), dm)
    k, look, _ = core.evaluate(dm, st, gen[0], 400, 1000, 10000, False, buf); return int(k), int(look)
gs, ks = [], []
for f in sorted(glob.glob(sys.argv[1] + "/gen_*.npz")):
    z = np.load(f); m = z["k"] >= 3000; gs.append(z["grids"][m]); ks.append(z["k"][m])
G = np.concatenate(gs); K = np.concatenate(ks)
rng = np.random.default_rng(3); idx = rng.choice(len(G), 120, replace=False)
bins = [-10**9, -1000, -300, -100, -30, -10, -1, 0, 1, 10, 30, 10**9]
names = ["<=-1000", "-999..-300", "-299..-100", "-99..-30", "-29..-10", "-9..-1", "0", "+1..+9", "+10..+29", "+30+"]
edges = [-10**9, -1000, -300, -100, -30, -10, 0, 1, 10, 30, 10**9]
cnt = np.zeros(len(names), dtype=np.int64); look_up = 0; tot = 0; per_grid_small = []
for i in idx:
    g = G[i].copy(); k0, l0 = score(g); small = 0
    for r in range(R):
        for c in range(C):
            old = g[r, c]
            for v in range(10):
                if v == old: continue
                g[r, c] = v; k1, l1 = score(g); g[r, c] = old
                d = k1 - k0; tot += 1
                b = np.searchsorted(edges, d, side="right") - 1
                cnt[min(b, len(names) - 1)] += 1
                if d == 0 and l1 > l0: look_up += 1
                if -30 < d: small += 1
    per_grid_small.append(small)
print(f"{len(idx)} grids x 1008 changes = {tot}  (base median K={int(np.median(K[idx]))})")
for n, c_ in zip(names, cnt): print(f"  dK {n:>11}: {c_:7d}  {c_ / tot:7.2%}")
print(f"  dK==0 but 'look' rises (smooth signal): {look_up} ({look_up / tot:.2%})")
print(f"  changes with dK > -30 per grid: median {int(np.median(per_grid_small))} of 1008")
