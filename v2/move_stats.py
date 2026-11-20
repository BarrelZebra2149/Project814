"""
Move statistics on our own K>=3000 grids:
  M1  single-cell change: fraction improving / neutral / worsening, and size of drops
  M2  row transplant (row r of grid B -> row r of grid A) vs replacing row r with random digits
      => does a "good row library" carry any signal beyond a random row?
Score used: (K, look) as K*1000+look, same as gen_data.py.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402

ROWS, COLS = core.ROWS, core.COLS


class Scorer:
    def __init__(self):
        self.dm = np.zeros((10, ROWS), dtype=np.int64)
        self.stamp = np.full(core.UPPER, -1, dtype=np.int64)
        self.buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
        self.gen = 0

    def __call__(self, g):
        self.gen += 1
        core.build_dmask(np.ascontiguousarray(g), self.dm)
        k, look, _ = core.evaluate(self.dm, self.stamp, self.gen, 400, 1000, 10000, False, self.buf)
        return int(k), int(look)


def load(d, kmin=3000):
    gs, ks = [], []
    for f in sorted(glob.glob(f"{d}/gen_*.npz")):
        z = np.load(f); m = z["k"] >= kmin
        gs.append(z["grids"][m]); ks.append(z["k"][m])
    return np.concatenate(gs), np.concatenate(ks)


def pct(a, q):
    return int(np.percentile(a, q))


def main(d):
    G, K = load(d)
    rng = np.random.default_rng(11)
    sc = Scorer()
    n = min(400, len(G))
    idx = rng.choice(len(G), size=n, replace=False)
    print(f"base grids: {n}  (K>=3000, median K={int(np.median(K[idx]))})")

    # M1: single-cell change, 30 random cells per grid
    imp = neu = wor = 0
    dK = []
    for i in idx:
        g = G[i]; k0, l0 = sc(g)
        for _ in range(30):
            r, c = rng.integers(ROWS), rng.integers(COLS)
            old = g[r, c]; new = (old + rng.integers(1, 10)) % 10
            g[r, c] = new
            k1, l1 = sc(g)
            g[r, c] = old
            v0, v1 = k0 * 1000 + l0, k1 * 1000 + l1
            if k1 > k0 or (k1 == k0 and l1 > l0):
                imp += 1
            elif k1 == k0 and l1 == l0:
                neu += 1
            else:
                wor += 1
            dK.append(k1 - k0)
    tot = imp + neu + wor
    dK = np.array(dK)
    print(f"\n[M1] single-cell change ({tot} trials)")
    print(f"  improve {imp/tot:.1%} | neutral(K and look equal) {neu/tot:.1%} | worse {wor/tot:.1%}")
    print(f"  K change: median {np.median(dK):.0f}, 10% {pct(dK,10)}, 1% {pct(dK,1)}, 99% {pct(dK,99)}, "
          f"P(dK<0)={np.mean(dK<0):.1%}, P(dK=0)={np.mean(dK==0):.1%}, P(dK>0)={np.mean(dK>0):.1%}")

    # M2: row transplant vs random row
    def run(kind):
        res = []
        for i in idx:
            A = G[i].copy(); k0, _ = sc(A)
            r = rng.integers(ROWS)
            if kind == "transplant":
                B = G[rng.choice(len(G))]
                A[r] = B[r]                       # same row position from another good grid
            elif kind == "transplant_any_pos":
                B = G[rng.choice(len(G))]
                A[r] = B[rng.integers(ROWS)]      # a good row, wrong position
            else:
                A[r] = rng.integers(0, 10, size=COLS)
            k1, _ = sc(A)
            res.append(k1 - k0)
        return np.array(res)

    print("\n[M2] replace one row (K change relative to the base grid)")
    for kind in ("transplant", "transplant_any_pos", "random_row"):
        x = run(kind)
        print(f"  {kind:20s}: median {np.median(x):8.0f}  90th pct {pct(x,90):6d}  "
              f"P(K>=base) {np.mean(x>=0):.1%}  P(K>base) {np.mean(x>0):.1%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_gen")
