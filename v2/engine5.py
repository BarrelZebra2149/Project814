"""
Chain-count engine, generalised to numbers 1..tmax (tmax >= 9999, e.g. 12000 for 5-digit coverage).
Same semantics as engine.py (which stays untouched: the running A/B imports it); differences:

  * `tmax` is a runtime parameter; walk length goes up to len(str(tmax)) (4 for 9999, 5 for 10000..99999)
  * numbers > tmax are ignored (never counted), so cnt/cntx are only (tmax+1) long
  * 5-digit walks whose leading digit (before AND after the change) exceeds tmax's leading part are
    skipped before any arithmetic (pruning: most 5-walks can never matter when tmax ~ 12000)
  * first_hole = tmax + 1 when every number 1..tmax is readable, and K = min(first_hole - 1, tmax)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

ROWS, COLS, NC = 8, 14, 112
HERE = Path(__file__).resolve().parent


def build_tables(ndig):
    nb = [[] for _ in range(NC)]
    for r in range(ROWS):
        for c in range(COLS):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
                        nb[r * COLS + c].append((r + dr) * COLS + c + dc)
    cur = [(x,) for x in range(NC)]
    allw = list(cur)
    for _ in range(ndig - 1):
        cur = [w + (y,) for w in cur for y in nb[w[-1]]]
        allw.extend(cur)
    n = len(allw)
    WALK = np.full((n, ndig), -1, dtype=np.int16)
    WLEN = np.zeros(n, dtype=np.int8)
    WD = np.full((n, ndig), -1, dtype=np.int16)
    WDN = np.zeros(n, dtype=np.int8)
    for i, w in enumerate(allw):
        WALK[i, :len(w)] = w
        WLEN[i] = len(w)
        d = list(dict.fromkeys(w))
        WD[i, :len(d)] = d
        WDN[i] = len(d)
    cells, gids = [], []
    for j in range(ndig):
        m = WD[:, j] >= 0
        cells.append(WD[m, j].astype(np.int64))
        gids.append(np.nonzero(m)[0])
    cells = np.concatenate(cells)
    gids = np.concatenate(gids)
    order = np.argsort(cells, kind="stable")
    t_wid = gids[order].astype(np.int32)
    t_off = np.concatenate([[0], np.cumsum(np.bincount(cells, minlength=NC))]).astype(np.int64)
    return WALK, WLEN, WD, WDN, t_off, t_wid


def load_tables(ndig):
    f = HERE / f"tables{ndig}.npz"
    if f.exists():
        z = np.load(f)
        return tuple(z[k] for k in ("WALK", "WLEN", "WD", "WDN", "t_off", "t_wid"))
    t = build_tables(ndig)
    np.savez(f, WALK=t[0], WLEN=t[1], WD=t[2], WDN=t[3], t_off=t[4], t_wid=t[5])
    return t


@njit(cache=True)
def init_state(grid, WALK, WLEN, WD, WDN, cnt, cntx, tmax, wt):
    cnt[:] = 0
    cntx[:, :] = 0
    for g in range(WALK.shape[0]):
        if grid[WALK[g, 0]] == 0:
            continue
        n = 0
        for j in range(WLEN[g]):
            n = n * 10 + grid[WALK[g, j]]
        if n > tmax:
            continue
        cnt[n] += 1
        for j in range(WDN[g]):
            cntx[n, WD[g, j]] += 1
    H = 0.0
    for n in range(1, tmax + 1):
        if cnt[n] == 0:
            H += wt[n]
    fh = 1
    while fh <= tmax and cnt[fh] > 0:
        fh += 1
    return H, fh


@njit(cache=True)
def eval_move(x, v, grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched, tmax, wt):
    nd = WALK.shape[1]
    p10 = 1
    for _ in range(nd - 1):
        p10 *= 10
    lm = tmax // p10                      # largest leading part of a max-length number that can be <= tmax
    nt = 0
    for e in range(t_off[x], t_off[x + 1]):
        g = t_wid[e]
        L = WLEN[g]
        c0 = WALK[g, 0]
        d0 = grid[c0]
        n0 = v if c0 == x else d0
        if L == nd and d0 > lm and n0 > lm:
            continue
        if d0 == 0 and n0 == 0:
            continue
        no = 0
        nn = 0
        for j in range(L):
            c = WALK[g, j]
            d = grid[c]
            no = no * 10 + d
            nn = nn * 10 + (v if c == x else d)
        if d0 != 0 and no <= tmax:
            if mark[no] != gen:
                mark[no] = gen
                touched[nt] = no
                nt += 1
            delta[no] -= 1
        if n0 != 0 and nn <= tmax:
            if mark[nn] != gen:
                mark[nn] = gen
                touched[nt] = nn
                nt += 1
            delta[nn] += 1
    dH = 0.0
    cand = tmax + 1
    for t in range(nt):
        n = touched[t]
        c0_ = cnt[n]
        c1 = c0_ + delta[n]
        if c0_ > 0 and c1 == 0:
            dH += wt[n]
            if n < cand:
                cand = n
        elif c0_ == 0 and c1 > 0:
            dH -= wt[n]
    nf = fh if fh < cand else cand
    while nf <= tmax and cnt[nf] + delta[nf] > 0:
        nf += 1
    return dH, nf, nt


@njit(cache=True)
def clear_deltas(delta, touched, nt):
    for t in range(nt):
        delta[touched[t]] = 0


@njit(cache=True)
def commit_deltas(x, v, grid, WALK, WLEN, WD, WDN, t_off, t_wid, cnt, cntx, delta, touched, nt, tmax):
    nd = WALK.shape[1]
    p10 = 1
    for _ in range(nd - 1):
        p10 *= 10
    lm = tmax // p10
    for e in range(t_off[x], t_off[x + 1]):
        g = t_wid[e]
        L = WLEN[g]
        c0 = WALK[g, 0]
        d0 = grid[c0]
        n0 = v if c0 == x else d0
        if L == nd and d0 > lm and n0 > lm:
            continue
        if d0 == 0 and n0 == 0:
            continue
        no = 0
        nn = 0
        for j in range(L):
            c = WALK[g, j]
            d = grid[c]
            no = no * 10 + d
            nn = nn * 10 + (v if c == x else d)
        if d0 != 0 and no <= tmax:
            for j in range(WDN[g]):
                cntx[no, WD[g, j]] -= 1
        if n0 != 0 and nn <= tmax:
            for j in range(WDN[g]):
                cntx[nn, WD[g, j]] += 1
    for t in range(nt):
        n = touched[t]
        cnt[n] += delta[n]
        delta[n] = 0
    grid[x] = v


@njit(cache=True)
def compute_cut(cnt, cntx, K, beta, p):
    crit = np.full(NC, K + 1, dtype=np.int64)
    spf = np.zeros(K + 1, dtype=np.bool_)
    F = 0.0
    for n in range(1, K + 1):
        c = cnt[n]
        if c <= 0:
            continue
        is_spf = False
        for x in range(NC):
            if cntx[n, x] == c:
                is_spf = True
                if n < crit[x]:
                    crit[x] = n
        spf[n] = is_spf
        F += (1.0 + (beta if is_spf else 0.0)) / (c ** p)
    free = 0
    for x in range(NC):
        if crit[x] > K:
            free += 1
    return crit, spf, free, F


@njit(cache=True)
def anneal(grid, cnt, cntx, H, fh, WALK, WLEN, WD, WDN, t_off, t_wid, delta, mark, gen, touched,
           n_steps, temp, seed, best_grid, tmax, wt):
    np.random.seed(seed)
    acc = 0
    bestK = min(fh - 1, tmax)
    for _ in range(n_steps):
        x = np.random.randint(0, NC)
        old = grid[x]
        v = np.random.randint(0, 9)
        if v >= old:
            v += 1
        gen += 1
        dH, nf, nt = eval_move(x, v, grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched, tmax, wt)
        if dH <= 0 or np.random.random() < np.exp(-dH / temp):
            commit_deltas(x, v, grid, WALK, WLEN, WD, WDN, t_off, t_wid, cnt, cntx, delta, touched, nt, tmax)
            H += dH
            fh = nf
            acc += 1
            if min(fh - 1, tmax) > bestK:
                bestK = min(fh - 1, tmax)
                best_grid[:] = grid
        else:
            clear_deltas(delta, touched, nt)
    return H, fh, gen, acc, bestK


class Engine:
    def __init__(self, grid=None, tmax=9999, a=0.0, core_mult=1.0, b5=0.0):
        assert tmax >= 9999
        self.tmax = int(tmax)
        self.NMAX = self.tmax + 1
        self.ndig = len(str(self.tmax))
        self.T = load_tables(self.ndig)
        t_off = self.T[4]
        self.cnt = np.zeros(self.NMAX, dtype=np.int32)
        self.cntx = np.zeros((self.NMAX, NC), dtype=np.int32)
        self.delta = np.zeros(self.NMAX, dtype=np.int32)
        self.mark = np.zeros(self.NMAX, dtype=np.int64)
        self.gen = 0
        self.touched = np.zeros(2 * int(np.diff(t_off).max()) + 16, dtype=np.int32)
        self.grid = np.zeros(NC, dtype=np.int8)
        ns = np.arange(self.NMAX, dtype=np.float64)
        w4 = core_mult * (1.0 + a * np.maximum(0.0, 1.0 - ns / 9999.0))          # n <= 9999
        span = max(self.tmax - 9999, 1)
        w5 = 1.0 + b5 * np.clip(1.0 - (ns - 9999.0) / span, 0.0, 1.0)             # n > 9999: lower numbers heavier
        self.wt = np.where(ns <= 9999, w4, w5)
        self.H = 0.0
        self.fh = 1
        if grid is not None:
            self.set_grid(grid)

    def set_grid(self, grid):
        self.grid[:] = np.asarray(grid, dtype=np.int8).reshape(-1)
        WALK, WLEN, WD, WDN, _, _ = self.T
        H, self.fh = init_state(self.grid, WALK, WLEN, WD, WDN, self.cnt, self.cntx, self.tmax, self.wt)
        self.H = float(H)

    @property
    def holes(self):
        """plain hole count in 1..tmax (H itself is the weighted energy)"""
        return int((self.cnt[1:] == 0).sum())

    @property
    def K(self):
        return min(self.fh - 1, self.tmax)

    def peek(self, x, v):
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        self.gen += 1
        dH, nf, nt = eval_move(x, v, self.grid, WALK, WLEN, t_off, t_wid, self.cnt, self.fh,
                               self.delta, self.mark, self.gen, self.touched, self.tmax, self.wt)
        clear_deltas(self.delta, self.touched, nt)
        return float(dH), min(int(nf) - 1, self.tmax)

    def commit(self, x, v):
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        self.gen += 1
        dH, nf, nt = eval_move(x, v, self.grid, WALK, WLEN, t_off, t_wid, self.cnt, self.fh,
                               self.delta, self.mark, self.gen, self.touched, self.tmax, self.wt)
        commit_deltas(x, v, self.grid, WALK, WLEN, WD, WDN, t_off, t_wid, self.cnt, self.cntx,
                      self.delta, self.touched, nt, self.tmax)
        self.H += float(dH)
        self.fh = int(nf)
        return float(dH)

    def fragility(self, beta=3.0, p=1.0):
        return compute_cut(self.cnt, self.cntx, self.K, beta, p)

    def anneal(self, n_steps, temp, seed, best_grid):
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        H, fh, gen, acc, bestK = anneal(self.grid, self.cnt, self.cntx, self.H, self.fh, WALK, WLEN, WD, WDN,
                                        t_off, t_wid, self.delta, self.mark, self.gen, self.touched,
                                        n_steps, float(temp), int(seed), best_grid, self.tmax, self.wt)
        self.H, self.fh, self.gen = float(H), int(fh), int(gen)
        return int(acc), int(bestK)
