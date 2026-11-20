"""
Single-point-of-failure (SPF) model check, on our own K>=3000 grids.

  cnt[n]     = number of directed walks (cell sequences, revisits allowed) that read n
  cntx[n,x]  = number of those walks that pass through cell x (a walk counts once per cell)
  x is a CUT CELL of n  <=>  cnt[n] > 0 and cntx[n,x] == cnt[n]   (every chain of n uses x)
  crit(x)    = smallest number n <= K whose cut cell is x  -> changing x kills n -> K drops to n-1
               (unless the new digit happens to create a fresh walk for n)

Checks:
  [1] walk model reproduces core814's K exactly  (first n with cnt==0 == K+1)
  [2] predicted K after changing x (ignoring new walks) vs. the real K for every (x, new digit)
  [3] stats: how many numbers <= K are single-point-of-failure, how many cells are 'free'
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402

R, C = core.ROWS, core.COLS
NC = R * C
NMAX = 10000


def build_walks():
    nb = [[] for _ in range(NC)]
    for r in range(R):
        for c in range(C):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < R and 0 <= c + dc < C:
                        nb[r * C + c].append((r + dr) * C + c + dc)
    out = {1: [[x] for x in range(NC)]}
    for L in (2, 3, 4):
        out[L] = [w + [y] for w in out[L - 1] for y in nb[w[-1]]]
    return [np.array(out[L], dtype=np.int32) for L in (1, 2, 3, 4)]


W1, W2, W3, W4 = build_walks()


@njit(cache=True)
def _tally(flat, W, cnt, cntx):
    L = W.shape[1]
    for i in range(W.shape[0]):
        if flat[W[i, 0]] == 0:
            continue                      # numbers have no leading zero
        n = 0
        for j in range(L):
            n = n * 10 + flat[W[i, j]]
        cnt[n] += 1
        for j in range(L):                # each distinct cell once
            x = W[i, j]
            seen = False
            for k in range(j):
                if W[i, k] == x:
                    seen = True
                    break
            if not seen:
                cntx[n, x] += 1


def tally(g):
    flat = np.ascontiguousarray(g).reshape(-1).astype(np.int64)
    cnt = np.zeros(NMAX, dtype=np.int32)
    cntx = np.zeros((NMAX, NC), dtype=np.int32)
    for W in (W1, W2, W3, W4):
        _tally(flat, W, cnt, cntx)
    return cnt, cntx


class Scorer:
    def __init__(self):
        self.dm = np.zeros((10, R), dtype=np.int64)
        self.st = np.full(core.UPPER, -1, dtype=np.int64)
        self.buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
        self.gen = 0

    def __call__(self, g):
        self.gen += 1
        core.build_dmask(np.ascontiguousarray(g), self.dm)
        return int(core.evaluate(self.dm, self.st, self.gen, 400, 1000, 10000, False, self.buf)[0])


def main(d):
    gs, ks = [], []
    for f in sorted(glob.glob(f"{d}/gen_*.npz")):
        z = np.load(f); m = z["k"] >= 3000
        gs.append(z["grids"][m]); ks.append(z["k"][m])
    G = np.concatenate(gs); Kall = np.concatenate(ks)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(G), size=60, replace=False)
    sc = Scorer()

    ok_k = 0
    eq = up = down = 0
    spf_frac, free_cells, crit_cells, cnt_med, spf_by_len = [], [], [], [], np.zeros(5)
    tot_by_len = np.zeros(5)
    for i in idx:
        g = G[i].copy()
        K = sc(g)
        cnt, cntx = tally(g)
        firsthole = int(np.argmax(cnt[1:] == 0)) + 1
        ok_k += (firsthole == K + 1)

        alive = np.arange(1, K + 1)
        cut = (cntx[1:K + 1] == cnt[1:K + 1, None]) & (cnt[1:K + 1, None] > 0)   # [K, NC]
        is_spf = cut.any(axis=1)
        spf_frac.append(is_spf.mean())
        for n, s in zip(alive, is_spf):
            L = len(str(n)); tot_by_len[L] += 1; spf_by_len[L] += s
        cnt_med.append(np.median(cnt[1000:K + 1]))
        # crit(x): smallest n <= K that dies if x changes (no new walks considered)
        pred = np.full(NC, K, dtype=np.int64)
        for x in range(NC):
            hit = np.nonzero(cut[:, x])[0]
            if len(hit):
                pred[x] = hit[0]           # index -> number hit[0]+1 dies -> K' = hit[0]
        free_cells.append(int((pred == K).sum()))
        crit_cells.append(int((pred < K - 300).sum()))

        flat = g.reshape(-1)
        for x in range(NC):
            old = flat[x]
            for v in range(10):
                if v == old:
                    continue
                flat[x] = v
                k2 = sc(g)
                flat[x] = old
                if k2 == pred[x]:
                    eq += 1
                elif k2 > pred[x]:
                    up += 1
                else:
                    down += 1

    tot = eq + up + down
    print(f"grids={len(idx)}  (median K={int(np.median(Kall[idx]))})")
    print(f"[1] walk model first hole == core814 K+1 : {ok_k}/{len(idx)}")
    print(f"[2] changes={tot}: real K' == predicted (cut-cell model) {eq/tot:.1%} | "
          f"real > predicted (new digit made a new chain) {up/tot:.1%} | real < predicted (MODEL WRONG) {down/tot:.2%}")
    print(f"[3] numbers <= K that are single-point-of-failure: mean {np.mean(spf_frac):.1%}")
    for L in (1, 2, 3, 4):
        if tot_by_len[L]:
            print(f"      {L}-digit: {spf_by_len[L] / tot_by_len[L]:.1%} SPF  (of {int(tot_by_len[L] / len(idx))} per grid)")
    print(f"    median chain count cnt[n] for 1000<=n<=K: {np.median(cnt_med):.0f}")
    print(f"    'free' cells (changing them kills nothing <= K): median {int(np.median(free_cells))} of 112")
    print(f"    'load-bearing' cells (changing kills a number < K-300): median {int(np.median(crit_cells))} of 112")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_gen")
