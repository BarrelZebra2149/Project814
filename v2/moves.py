"""
Moves M1..M4 on top of engine.Engine (PLAN v3.1, section 6).  Every move is evaluated EXACTLY by the
engine; a move is applied for real and reverted exactly (commit of the inverse) if rejected.

  fixes_for(n)   (x, v) pairs such that changing cell x to digit v creates a NEW chain for number n,
                 with the number of distinct walks supporting each pair ("overlap strength").
  M1 scan_move   rejection-free heat bath over all 1008 single-cell changes (+ a 'stay' option)
  M2 reinforce   pick a single-point-of-failure number, add a chain to it via a one-cell fix
  M3 extend      fix the first hole with a one-cell fix that overlaps existing chains
  M4 repair      break-and-repair: apply a fix that breaks other numbers, then repair each newly
                 broken number with its own one-cell fix, up to `depth` repairs; accept the whole
                 bundle by energy, else revert all.
"""
from __future__ import annotations

import numpy as np
from numba import njit

import engine as eg

NC = eg.NC


# --------------------------------------------------------------------------- numba kernels
@njit(cache=True)
def fixes_for(digits, L, grid, WALK, lo, hi, fixcnt):
    """fixcnt[x, v] = #walks of length L that read digits except for exactly ONE distinct cell x
    which would need digit v (and every other cell already matches). 0 for readable-already walks."""
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
def scan_all(grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched, out_dH, out_K):
    for x in range(NC):
        for v in range(10):
            if v == grid[x]:
                out_dH[x, v] = 0
                out_K[x, v] = fh - 1
                continue
            gen += 1
            dH, nf, nt = eg.eval_move(x, v, grid, WALK, WLEN, t_off, t_wid, cnt, fh, delta, mark, gen, touched)
            eg.clear_deltas(delta, touched, nt)
            out_dH[x, v] = dH
            out_K[x, v] = nf - 1
    return gen


# --------------------------------------------------------------------------- searcher
class Searcher:
    def __init__(self, grid, seed=0, temp=3.0, aux=False, b=5.0, c=0.5, beta=3.0, p=1.0):
        self.e = eg.Engine(grid)
        self.rng = np.random.default_rng(seed)
        self.temp = float(temp)
        self.aux = aux
        self.b, self.c, self.beta, self.p = b, c, beta, p
        WALK, WLEN, WD, WDN, t_off, t_wid = self.e.T
        self.lo = {L: int(np.searchsorted(WLEN, L, side="left")) for L in (1, 2, 3, 4)}
        self.hi = {L: int(np.searchsorted(WLEN, L, side="right")) for L in (1, 2, 3, 4)}
        self.fixcnt = np.zeros((NC, 10), dtype=np.int32)
        self.dHm = np.zeros((NC, 10), dtype=np.int64)
        self.Km = np.zeros((NC, 10), dtype=np.int64)
        self.stats = {k: {"tried": 0, "accepted": 0, "improved": 0} for k in ("M1", "M2", "M3", "M4")}
        self.m4_depth = {}          # depth -> [tried, accepted_with_dE<=0]
        self.best_K = self.e.K
        self.best_grid = self.e.grid.copy()

    # ---- energy
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

    # ---- helpers
    def candidates(self, n):
        """list of (x, v, support) for number n (n >= 1)."""
        s = str(n)
        L = len(s)
        digits = np.array([int(ch) for ch in s], dtype=np.int64)
        WALK = self.e.T[0]
        fixes_for(digits, L, self.e.grid, WALK, self.lo[L], self.hi[L], self.fixcnt)
        xs, vs = np.nonzero(self.fixcnt)
        return [(int(x), int(v), int(self.fixcnt[x, v])) for x, v in zip(xs, vs)]

    def _revert(self, stack):
        for x, old in reversed(stack):
            self.e.commit(x, old)

    # ---- M1
    def scan_move(self):
        e = self.e
        WALK, WLEN, WD, WDN, t_off, t_wid = e.T
        e.gen = int(scan_all(e.grid, WALK, WLEN, t_off, t_wid, e.cnt, e.fh, e.delta, e.mark, e.gen,
                             e.touched, self.dHm, self.Km))
        w = np.exp(-self.dHm.astype(np.float64) / self.temp)
        w[np.arange(NC)[:, None], e.grid[:, None].astype(np.int64)] = 0.0      # no-op entries
        w_stay = 1.0
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

    # ---- M3
    def extend(self):
        e = self.e
        self.stats["M3"]["tried"] += 1
        if e.fh >= eg.NMAX:
            return False
        cands = self.candidates(e.fh)
        if not cands:
            return False
        scored = []
        for x, v, sup in cands:
            dH, Kp = e.peek(x, v)
            scored.append((dH, -sup, x, v, Kp))
        scored.sort()
        dH, _, x, v, Kp = scored[0]
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

    # ---- M2
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

    # ---- M4
    def repair(self, depth=4):
        e = self.e
        self.stats["M4"]["tried"] += 1
        if e.fh >= eg.NMAX:
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

    # ---- fast random-proposal Metropolis chunk (jitted); the cheap base that the structured moves are mixed into
    def rand_chunk(self, n=2000):
        if not hasattr(self, "_bg"):
            self._bg = np.zeros(NC, dtype=np.int8)
        acc, bk = self.e.anneal(n, self.temp, int(self.rng.integers(1, 2**31 - 1)), self._bg)
        if bk > self.best_K:
            self.best_K = int(bk)
            self.best_grid = self._bg.copy()
        self._track()
        return acc > 0

    # ---- driver
    def step(self, mix):
        r = self.rng.random()
        acc = 0.0
        for name, p in mix:
            acc += p
            if r < acc:
                return getattr(self, name)()
        return getattr(self, mix[-1][0])()
