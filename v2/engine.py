"""
Incremental chain-count engine (PLAN v3.1, section 4).

State (all exact, numbers 1..9999 only; 5-digit numbers are out of scope, PLAN Q5):
  grid[112]       digits
  cnt[n]          number of directed walks (cell sequences, revisits allowed, king moves,
                  length = number of digits of n, no leading zero) that read n
  cntx[n, x]      how many of those walks pass through cell x (a walk counts once per distinct cell)
  H               holes = #{ n in 1..9999 : cnt[n] == 0 }
  first_hole      smallest n with cnt[n] == 0  (K = first_hole - 1, capped at 9999)

Two-tier cost model (measured in test_engine.py):
  * eval_move(x, v)   : cnt/H/first_hole after the change, WITHOUT touching state. Cheap; used for every candidate.
  * commit(x, v)      : applies the change AND keeps cntx exact. Only paid for accepted moves.
Fragility (cut cells, SPF numbers, crit, F) is computed on demand from cntx (compute_cut).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

ROWS, COLS, NC = 8, 14, 112
NMAX = 10000
TABLE_FILE = Path(__file__).resolve().parent / "tables.npz"


# --------------------------------------------------------------------------- tables
def build_tables():
    nb = [[] for _ in range(NC)]
    for r in range(ROWS):
        for c in range(COLS):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if (dr or dc) and 0 <= r + dr < ROWS and 0 <= c + dc < COLS:
                        nb[r * COLS + c].append((r + dr) * COLS + c + dc)
    cur = [[x] for x in range(NC)]
    allw = list(cur)
    for _ in range(3):
        cur = [w + [y] for w in cur for y in nb[w[-1]]]
        allw.extend(cur)
    n = len(allw)
    WALK = np.full((n, 4), -1, dtype=np.int16)
    WLEN = np.zeros(n, dtype=np.int8)
    WD = np.full((n, 4), -1, dtype=np.int16)
    WDN = np.zeros(n, dtype=np.int8)
    for i, w in enumerate(allw):
        WALK[i, :len(w)] = w
        WLEN[i] = len(w)
        d = []
        for c in w:
            if c not in d:
                d.append(c)
        WD[i, :len(d)] = d
        WDN[i] = len(d)
    cells, gids = [], []
    for j in range(4):
        m = WD[:, j] >= 0
        cells.append(WD[m, j].astype(np.int64))
        gids.append(np.nonzero(m)[0])
    cells = np.concatenate(cells); gids = np.concatenate(gids)
    order = np.argsort(cells, kind="stable")
    t_wid = gids[order].astype(np.int32)
    t_off = np.concatenate([[0], np.cumsum(np.bincount(cells, minlength=NC))]).astype(np.int64)
    return WALK, WLEN, WD, WDN, t_off, t_wid


def load_tables():
    if TABLE_FILE.exists():
        z = np.load(TABLE_FILE)
        return tuple(z[k] for k in ("WALK", "WLEN", "WD", "WDN", "t_off", "t_wid"))
    t = build_tables()
    np.savez(TABLE_FILE, WALK=t[0], WLEN=t[1], WD=t[2], WDN=t[3], t_off=t[4], t_wid=t[5])
    return t


# --------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def init_state(grid, WALK, WLEN, WD, WDN, cnt, cntx):
    cnt[:] = 0
    cntx[:, :] = 0
    for g in range(WALK.shape[0]):
        if grid[WALK[g, 0]] == 0:
            continue
        n = 0
        for j in range(WLEN[g]):
            n = n * 10 + grid[WALK[g, j]]
        cnt[n] += 1
        for j in range(WDN[g]):
            cntx[n, WD[g, j]] += 1
    H = 0
    for n in range(1, NMAX):
        if cnt[n] == 0:
            H += 1
    fh = 1
    while fh < NMAX and cnt[fh] > 0:
        fh += 1
    return H, fh


@njit(cache=True)
def eval_move(x, v, grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched):
    """Fill delta[] for the change grid[x] -> v. Returns (dH, new_first_hole, n_touched).
    Leaves delta/touched populated so commit_deltas() can apply them; call
    clear_deltas() to discard."""
    nt = 0
    for e in range(t_off[x], t_off[x + 1]):
        g = t_wid[e]
        L = WLEN[g]
        no = 0
        nn = 0
        for j in range(L):
            c = WALK[g, j]
            d = grid[c]
            no = no * 10 + d
            nn = nn * 10 + (v if c == x else d)
        c0 = WALK[g, 0]
        d0 = grid[c0]
        if d0 != 0:
            if mark[no] != gen:
                mark[no] = gen
                touched[nt] = no
                nt += 1
            delta[no] -= 1
        if (v if c0 == x else d0) != 0:
            if mark[nn] != gen:
                mark[nn] = gen
                touched[nt] = nn
                nt += 1
            delta[nn] += 1
    dH = 0
    cand = NMAX
    for t in range(nt):
        n = touched[t]
        c0_ = cnt[n]
        c1 = c0_ + delta[n]
        if c0_ > 0 and c1 == 0:
            dH += 1
            if n < cand:
                cand = n
        elif c0_ == 0 and c1 > 0:
            dH -= 1
    nf = fh if fh < cand else cand
    while nf < NMAX and cnt[nf] + delta[nf] > 0:
        nf += 1
    return dH, nf, nt


@njit(cache=True)
def clear_deltas(delta, touched, nt):
    for t in range(nt):
        delta[touched[t]] = 0


@njit(cache=True)
def commit_deltas(x, v, grid, WALK, WLEN, WD, WDN, t_off, t_wid, cnt, cntx, delta, touched, nt):
    """Apply a previously evaluated move: cnt, cntx, grid. (H/first_hole are applied by the caller.)"""
    for e in range(t_off[x], t_off[x + 1]):
        g = t_wid[e]
        L = WLEN[g]
        no = 0
        nn = 0
        for j in range(L):
            c = WALK[g, j]
            d = grid[c]
            no = no * 10 + d
            nn = nn * 10 + (v if c == x else d)
        c0 = WALK[g, 0]
        d0 = grid[c0]
        if d0 != 0:
            for j in range(WDN[g]):
                cntx[no, WD[g, j]] -= 1
        if (v if c0 == x else d0) != 0:
            for j in range(WDN[g]):
                cntx[nn, WD[g, j]] += 1
    for t in range(nt):
        n = touched[t]
        cnt[n] += delta[n]
        delta[n] = 0
    grid[x] = v


@njit(cache=True)
def compute_cut(cnt, cntx, K, beta, p):
    """Fragility over numbers 1..K.
    Returns (crit[112], spf[K+1], free_cells, F)
      crit[x] = smallest n <= K for which cell x is a cut cell (all chains of n pass through x); K+1 if none
      spf[n]  = n has at least one cut cell
      F       = sum_n w(n), w(n) = (1 + beta*spf[n]) / cnt[n]**p   (PLAN 5.2)"""
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
           n_steps, temp, seed, best_grid):
    """Plain single-cell annealing on E = H (hole count). Speed benchmark / baseline arm."""
    np.random.seed(seed)
    acc = 0
    bestK = fh - 1
    for _ in range(n_steps):
        x = np.random.randint(0, NC)
        old = grid[x]
        v = np.random.randint(0, 9)
        if v >= old:
            v += 1
        gen += 1
        dH, nf, nt = eval_move(x, v, grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched)
        if dH <= 0 or np.random.random() < np.exp(-dH / temp):
            commit_deltas(x, v, grid, WALK, WLEN, WD, WDN, t_off, t_wid, cnt, cntx, delta, touched, nt)
            H += dH
            fh = nf
            acc += 1
            if fh - 1 > bestK:
                bestK = fh - 1
                best_grid[:] = grid
        else:
            clear_deltas(delta, touched, nt)
    return H, fh, gen, acc, bestK


# --------------------------------------------------------------------------- python wrapper
class Engine:
    def __init__(self, grid=None):
        self.T = load_tables()
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        self.cnt = np.zeros(NMAX, dtype=np.int32)
        self.cntx = np.zeros((NMAX, NC), dtype=np.int32)
        self.delta = np.zeros(NMAX, dtype=np.int32)
        self.mark = np.zeros(NMAX, dtype=np.int64)
        self.gen = 0
        self.touched = np.zeros(2 * int(np.diff(t_off).max()) + 16, dtype=np.int32)
        self.grid = np.zeros(NC, dtype=np.int8)
        self.H = 0
        self.fh = 1
        if grid is not None:
            self.set_grid(grid)

    def set_grid(self, grid):
        self.grid[:] = np.asarray(grid, dtype=np.int8).reshape(-1)
        WALK, WLEN, WD, WDN, _, _ = self.T
        self.H, self.fh = init_state(self.grid, WALK, WLEN, WD, WDN, self.cnt, self.cntx)

    @property
    def K(self):
        return min(self.fh - 1, NMAX - 1)

    def peek(self, x, v):
        """-> (dH, K_after) without changing state."""
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        self.gen += 1
        dH, nf, nt = eval_move(x, v, self.grid, WALK, WLEN, t_off, t_wid, self.cnt, self.fh,
                               self.delta, self.mark, self.gen, self.touched)
        clear_deltas(self.delta, self.touched, nt)
        return int(dH), min(int(nf) - 1, NMAX - 1)

    def commit(self, x, v):
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        self.gen += 1
        dH, nf, nt = eval_move(x, v, self.grid, WALK, WLEN, t_off, t_wid, self.cnt, self.fh,
                               self.delta, self.mark, self.gen, self.touched)
        commit_deltas(x, v, self.grid, WALK, WLEN, WD, WDN, t_off, t_wid, self.cnt, self.cntx,
                      self.delta, self.touched, nt)
        self.H += int(dH)
        self.fh = int(nf)
        return int(dH)

    def fragility(self, beta=3.0, p=1.0):
        return compute_cut(self.cnt, self.cntx, self.K, beta, p)

    def anneal(self, n_steps, temp, seed, best_grid):
        WALK, WLEN, WD, WDN, t_off, t_wid = self.T
        H, fh, gen, acc, bestK = anneal(self.grid, self.cnt, self.cntx, self.H, self.fh, WALK, WLEN, WD, WDN,
                                        t_off, t_wid, self.delta, self.mark, self.gen, self.touched,
                                        n_steps, float(temp), int(seed), best_grid)
        self.H, self.fh, self.gen = int(H), int(fh), int(gen)
        return int(acc), int(bestK)
