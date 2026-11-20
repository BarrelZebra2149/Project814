"""
moves.py generalised to engine5 (numbers 1..tmax, up to 5 digits).  Logic is identical to moves.py
(which stays untouched because the running A/B imports it); only the engine API differs.
"""
from __future__ import annotations

import numpy as np
from numba import njit

import engine5 as eg

NC = eg.NC


@njit(cache=True)
def fixes_for(digits, L, grid, WALK, lo, hi, fixcnt):
    fixcnt[:, :] = 0
    for g in range(lo, hi):
        nmis = 0
        fx = -1
        fv = -1
        ok = True
        for j in range(L):
            c = WALK[g, j]
            need = digits[j]
            first = True
            for k in range(j):
                if WALK[g, k] == c:
                    first = False
                    if digits[k] != need:
                        ok = False
                    break
            if not ok:
                break
            if first and grid[c] != need:
                nmis += 1
                fx = c
                fv = need
                if nmis > 1:
                    ok = False
                    break
        if ok and nmis == 1:
            fixcnt[fx, fv] += 1


@njit(cache=True)
def scan_all(grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched, out_dH, out_K, tmax, wt):
    for x in range(NC):
        for v in range(10):
            if v == grid[x]:
                out_dH[x, v] = 0
                out_K[x, v] = min(fh - 1, tmax)
                continue
            gen += 1
            dH, nf, nt = eg.eval_move(x, v, grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched, tmax, wt)
            eg.clear_deltas(delta, touched, nt)
            out_dH[x, v] = dH
            out_K[x, v] = min(nf - 1, tmax)
    return gen


class Searcher:
    def __init__(self, grid, seed=0, temp=3.0, aux=False, b=5.0, c=0.5, beta=3.0, p=1.0, tmax=9999, a=0.0, core_mult=1.0, b5=0.0):
        self.e = eg.Engine(grid, tmax=tmax, a=a, core_mult=core_mult, b5=b5)
        self.rng = np.random.default_rng(seed)
        self.temp = float(temp)
        self.aux = aux
        self.b, self.c, self.beta, self.p = b, c, beta, p
        WALK, WLEN, WD, WDN, t_off, t_wid = self.e.T
        Ls = range(1, self.e.ndig + 1)
        self.lo = {L: int(np.searchsorted(WLEN, L, side="left")) for L in Ls}
        self.hi = {L: int(np.searchsorted(WLEN, L, side="right")) for L in Ls}
        self.fixcnt = np.zeros((NC, 10), dtype=np.int32)
        self.dHm = np.zeros((NC, 10), dtype=np.float64)
        self.Km = np.zeros((NC, 10), dtype=np.int64)
        self.stats = {k: {"tried": 0, "accepted": 0, "improved": 0} for k in ("M1", "M2", "M3", "M4")}
        self.m4_depth = {}
        self.best_K = self.e.K
        self.best_grid = self.e.grid.copy()

    def energy(self):
        e = self.e
        if not self.aux:
            return float(e.H)
        crit, spf, free, F = e.fragility(self.beta, self.p)
        K = max(e.K, 1)
        return float(e.H) + self.b * F / K + self.c * (NC - free)

    def _track(self):
        if self.e.K > self.best_K:
            self.best_K = self.e.K
            self.best_grid = self.e.grid.copy()

    def _accept(self, dE):
        return dE <= 0 or self.rng.random() < np.exp(-dE / self.temp)

    def candidates(self, n):
        s = str(n)
        L = len(s)
        digits = np.array([int(ch) for ch in s], dtype=np.int64)
        fixes_for(digits, L, self.e.grid, self.e.T[0], self.lo[L], self.hi[L], self.fixcnt)
        xs, vs = np.nonzero(self.fixcnt)
        return [(int(x), int(v), int(self.fixcnt[x, v])) for x, v in zip(xs, vs)]

    def _revert(self, stack):
        for x, old in reversed(stack):
            self.e.commit(x, old)

    def scan_move(self):
        e = self.e
        WALK, WLEN, WD, WDN, t_off, t_wid = e.T
        e.gen = int(scan_all(e.grid, WALK, WLEN, t_off, t_wid, e.cnt, e.fh, e.delta, e.mark, e.gen,
                             e.touched, self.dHm, self.Km, e.tmax, e.wt))
        # numerically stable softmax over {all single-cell changes} + {stay}: logits = -dE/temp, stay logit = 0.
        # (the naive exp(-dE/temp) overflows to inf -> NaN probabilities when temp is small and dE very negative)
        logits = -self.dHm.astype(np.float64) / self.temp
        logits[np.arange(NC)[:, None], e.grid[:, None].astype(np.int64)] = -np.inf     # no-op entries
        m = max(float(logits.max()), 0.0)
        w = np.exp(logits - m)
        w_stay = float(np.exp(0.0 - m))
        tot = w.sum() + w_stay
        self.stats["M1"]["tried"] += 1
        if self.rng.random() * tot < w_stay:
            return False
        flat = w.reshape(-1)
        i = int(self.rng.choice(flat.size, p=flat / flat.sum()))
        x, v = divmod(i, 10)
        dH = e.commit(x, v)
        self.stats["M1"]["accepted"] += 1
        self.stats["M1"]["improved"] += dH < 0
        self._track()
        return True

    def extend(self):
        e = self.e
        self.stats["M3"]["tried"] += 1
        if e.fh > e.tmax:
            return False
        cands = self.candidates(e.fh)
        if not cands:
            return False
        scored = sorted(((e.peek(x, v)[0], -sup, x, v) for x, v, sup in cands))
        dH, _, x, v = scored[0]
        E0 = self.energy() if self.aux else float(e.H)
        old = int(e.grid[x])
        e.commit(x, v)
        dE = self.energy() - E0
        if self._accept(dE):
            self.stats["M3"]["accepted"] += 1
            self.stats["M3"]["improved"] += dE < 0
            self._track()
            return True
        e.commit(x, old)
        return False

    def reinforce(self, n_try=4):
        e = self.e
        self.stats["M2"]["tried"] += 1
        K = e.K
        if K < 10:
            return False
        crit, spf, free, F = e.fragility(self.beta, self.p)
        ns = np.nonzero(spf[1:])[0] + 1
        if len(ns) == 0:
            return False
        wts = 1.0 / e.cnt[ns].astype(np.float64)
        n = int(self.rng.choice(ns, p=wts / wts.sum()))
        cands = self.candidates(n)
        if not cands:
            return False
        scored = sorted(((e.peek(x, v)[0], -sup, x, v) for x, v, sup in cands))[:n_try]
        E0 = self.energy()
        best = None
        for dH, _, x, v in scored:
            old = int(e.grid[x])
            e.commit(x, v)
            dE = self.energy() - E0
            e.commit(x, old)
            if best is None or dE < best[0]:
                best = (dE, x, v)
        dE, x, v = best
        if self._accept(dE):
            e.commit(x, v)
            self.stats["M2"]["accepted"] += 1
            self.stats["M2"]["improved"] += dE < 0
            self._track()
            return True
        return False

    def repair(self, depth=4):
        e = self.e
        self.stats["M4"]["tried"] += 1
        if e.fh > e.tmax:
            return False
        cands = self.candidates(e.fh)
        if not cands:
            return False
        wts = np.array([c[2] for c in cands], dtype=np.float64)
        x, v, _ = cands[int(self.rng.choice(len(cands), p=wts / wts.sum()))]
        E0 = self.energy()
        holes0 = set(np.nonzero(e.cnt[1:] == 0)[0] + 1)
        touched = {x}
        stack = [(x, int(e.grid[x]))]
        e.commit(x, v)
        used = 0
        for d in range(depth):
            holes = set(np.nonzero(e.cnt[1:] == 0)[0] + 1)
            broken = sorted(holes - holes0)
            if not broken:
                break
            n = broken[0]
            cs = [c for c in self.candidates(n) if c[0] not in touched]
            if not cs:
                break
            cs.sort(key=lambda c: -c[2])
            best = min(((e.peek(cx, cv)[0], -sup, cx, cv) for cx, cv, sup in cs[:48]))
            _, _, cx, cv = best
            touched.add(cx)
            stack.append((cx, int(e.grid[cx])))
            e.commit(cx, cv)
            used += 1
        dE = self.energy() - E0
        rec = self.m4_depth.setdefault(used, [0, 0])
        rec[0] += 1
        if self._accept(dE):
            rec[1] += dE <= 0
            self.stats["M4"]["accepted"] += 1
            self.stats["M4"]["improved"] += dE < 0
            self._track()
            return True
        self._revert(stack)
        return False

    def rand_chunk(self, n=2000):
        if not hasattr(self, "_bg"):
            self._bg = np.zeros(NC, dtype=np.int8)
        acc, bk = self.e.anneal(n, self.temp, int(self.rng.integers(1, 2**31 - 1)), self._bg)
        if bk > self.best_K:
            self.best_K = int(bk)
            self.best_grid = self._bg.copy()
        self._track()
        return acc > 0
