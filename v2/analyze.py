"""
Statistics for PLAN.md sections 2.3 / 4, computed ONLY on data we generated
ourselves (gen_data.py output). Nothing from the old corpus.

    python analyze.py data_gen > report.txt
"""
from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "scoring"))
import core814 as core  # noqa: E402

ROWS, COLS = core.ROWS, core.COLS
DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load(d):
    gs, ks, ls, ss, rs = [], [], [], [], []
    for f in sorted(glob.glob(f"{d}/gen_*.npz")):
        z = np.load(f)
        gs.append(z["grids"]); ks.append(z["k"]); ls.append(z["look"])
        ss.append(z["start"]); rs.append(z["run"])
    return (np.concatenate(gs), np.concatenate(ks), np.concatenate(ls),
            np.concatenate(ss), np.concatenate(rs))


class Scorer:
    def __init__(self):
        self.dm = np.zeros((10, ROWS), dtype=np.int64)
        self.stamp = np.full(core.UPPER, -1, dtype=np.int64)
        self.buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
        self.gen = 0

    def __call__(self, g):
        self.gen += 1
        core.build_dmask(np.ascontiguousarray(g), self.dm)
        return core.evaluate(self.dm, self.stamp, self.gen, 400, 1000, 10000, False, self.buf)


def adjacency(g):
    """10x10 symmetric count of king-adjacent cell pairs holding digits (a,b)."""
    A = np.zeros((10, 10), dtype=np.int32)
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in DELTAS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    A[g[r, c], g[nr, nc]] += 1   # each undirected pair counted from both ends
    return A


def features(g):
    A = adjacency(g)
    iu = np.triu_indices(10)
    pair = A[iu].astype(np.float32)                 # 55 values (a<=b)
    freq = np.bincount(g.ravel(), minlength=10).astype(np.float32)
    present = (A > 0)
    # 3-chains a-b-c over the *digit* adjacency graph (walk may reuse cells)
    P = present.astype(np.int32)
    tri = float((P @ P).sum())
    deg = P.sum(1).astype(np.float32)
    return np.concatenate([pair, freq, deg, [P.sum(), tri, (A == 0).sum()]]).astype(np.float32)


def bottleneck_kind(g, n):
    """Why can't n be formed?  'pair' = some consecutive digit pair never
    adjacent anywhere; 'dup' = n needs a path through a repeated digit pair
    longer than the grid supports; else 'path'."""
    A = adjacency(g)
    s = str(n)
    for a, b in zip(s, s[1:]):
        if A[int(a), int(b)] == 0:
            return "pair"
    if any(d not in set(map(str, np.unique(g))) for d in s):
        return "digit"
    return "path"


def main(d):
    G, K, L, S, R = load(d)
    print(f"records={len(G)}  runs={len(np.unique(R))}  max K={K.max()}")
    for name, m in (("zero-start", S == 0), ("random-start", S == 1)):
        print(f"  {name}: n={m.sum()} median K={np.median(K[m]):.0f} 99%={np.percentile(K[m],99):.0f} max={K[m].max()}")

    # ---------------------------------------------------------------- A. symmetry
    print("\n[A] symmetry invariance (PLAN 2.3): score of 4 variants must be identical")
    sc = Scorer()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(G), size=min(500, len(G)), replace=False)
    bad = 0
    for i in idx:
        g = G[i]
        vs = [g, g[:, ::-1], g[::-1, :], g[::-1, ::-1]]
        ks = {sc(v)[0] for v in vs}
        bad += len(ks) != 1
    print(f"  tested {len(idx)} grids -> {bad} with differing scores")
    # a transpose-like / digit-relabel control: relabeling digits SHOULD change score
    ch = 0
    for i in idx[:200]:
        g = G[i]; p = rng.permutation(10).astype(np.uint8)
        ch += sc(p[g])[0] != sc(g)[0]
    print(f"  control: random digit relabeling changed the score in {ch}/200 grids (expected: most)")

    # ---------------------------------------------------------------- B. bottlenecks
    print("\n[B] bottleneck: which number is K+1 (the first one that fails)?")
    hi = K >= np.percentile(K, 50)
    nd = Counter(len(str(k + 1)) for k in K[hi])
    print("  digit-length of K+1 (upper half of records):", dict(sorted(nd.items())))
    top = Counter(int(k + 1) for k in K[hi]).most_common(8)
    print("  most common failing numbers:", top)
    last_digit = Counter(int(str(k + 1)[-1]) for k in K[hi])
    first_digit = Counter(int(str(k + 1)[0]) for k in K[hi])
    print("  first digit of K+1:", dict(sorted(first_digit.items())))
    print("  last digit of K+1: ", dict(sorted(last_digit.items())))
    kinds = Counter()
    sub = rng.choice(np.where(hi)[0], size=min(800, hi.sum()), replace=False)
    for i in sub:
        kinds[bottleneck_kind(G[i], int(K[i]) + 1)] += 1
    print("  why K+1 fails (pair=missing adjacency, digit=missing digit, path=pairs ok but no walk):", dict(kinds))

    # ---------------------------------------------------------------- C. predictability
    print("\n[C] can features predict K?  (group split by trajectory => no leakage)")
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.stack([features(g) for g in G])
    runs = np.unique(R)
    rr = np.random.default_rng(1)
    test_runs = set(rr.choice(runs, size=max(1, len(runs) // 5), replace=False))
    te = np.array([r in test_runs for r in R]); tr = ~te
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0)
    m.fit(X[tr], K[tr])
    pred = m.predict(X[te])
    rho = spearmanr(pred, K[te]).statistic
    print(f"  all records:            Spearman={rho:.3f}  (n_test={te.sum()})")
    for lo in (np.percentile(K, 50), np.percentile(K, 80), np.percentile(K, 95)):
        mm = te & (K >= lo)
        if mm.sum() > 30:
            print(f"  only K >= {lo:5.0f}:        Spearman={spearmanr(m.predict(X[mm]), K[mm]).statistic:.3f}  (n={mm.sum()})")
    # within a single trajectory step: does the model tell better from worse
    # neighbours?  (this is what a generator would actually need)
    gain = []
    for r in list(test_runs)[:60]:
        ii = np.where(R == r)[0]
        if len(ii) > 20:
            gain.append(spearmanr(m.predict(X[ii]), K[ii]).statistic)
    print(f"  within-trajectory Spearman (mean over {len(gain)} test runs): {np.nanmean(gain):.3f}")
    imp_idx = np.argsort(-np.abs(np.corrcoef(X[te].T, K[te])[-1, :-1]))[:8]
    names = [f"pair{a}{b}" for a, b in zip(*np.triu_indices(10))] + [f"freq{d}" for d in range(10)] + \
            [f"deg{d}" for d in range(10)] + ["n_pairs_present", "tri", "n_pairs_absent"]
    cors = np.corrcoef(X[te].T, K[te])[-1, :-1]
    print("  top |corr| features:", [(names[i], round(float(cors[i]), 3)) for i in imp_idx])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data_gen")
