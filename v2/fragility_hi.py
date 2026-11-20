"""
READ-ONLY measurement (user-approved, Q1): single-point-of-failure structure of
HIGH-score grids. Old corpus grids are used ONLY to compute these statistics --
never as search seeds or training data.

    python fragility_hi.py ../data 6000
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragility as fr  # noqa: E402

R, C = fr.R, fr.C


def read_grids(d):
    out, seen = [], set()
    for f in sorted(glob.glob(f"{d}/*.txt")):
        rows = []
        for line in Path(f).read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            if len(s) == C and s.isdigit():
                rows.append([int(ch) for ch in s])
                if len(rows) == R:
                    g = np.array(rows, dtype=np.uint8)
                    k = g.tobytes()
                    if k not in seen:
                        seen.add(k); out.append(g)
                    rows = []
            else:
                rows = []
    return out


def analyze(g, sc):
    K = sc(g)
    cnt, cntx = fr.tally(g)
    cut = (cntx[1:K + 1] == cnt[1:K + 1, None]) & (cnt[1:K + 1, None] > 0)
    is_spf = cut.any(axis=1)
    pred = np.full(fr.NC, K, dtype=np.int64)
    for x in range(fr.NC):
        hit = np.nonzero(cut[:, x])[0]
        if len(hit):
            pred[x] = hit[0]
    # real single-cell change outcomes
    flat = g.reshape(-1); small = neu = tot = up = 0
    for x in range(fr.NC):
        old = flat[x]
        for v in range(10):
            if v == old:
                continue
            flat[x] = v; k2 = sc(g); flat[x] = old
            tot += 1
            d = k2 - K
            if d > 0: up += 1
            elif d == 0: neu += 1
            elif d > -30: small += 1
    nums4 = np.arange(1000, K + 1)
    return dict(K=K, spf=is_spf.mean(), spf4=is_spf[999:K].mean() if K >= 1000 else 0.0,
                cnt_med=float(np.median(cnt[1000:K + 1])), cnt1=float((cnt[1000:K + 1] == 1).mean()),
                free=int((pred == K).sum()), load=int((pred < K - 300).sum()),
                up=up / tot, neu=neu / tot, small=small / tot)


def main(d, kmin):
    grids = read_grids(d)
    sc = fr.Scorer()
    scored = sorted(((sc(g), i) for i, g in enumerate(grids)), reverse=True)
    sel = [grids[i] for k, i in scored if k >= kmin]
    print(f"read {len(grids)} unique grids; K>={kmin}: {len(sel)}  (max K={scored[0][0]})")
    if not sel:
        return
    rows = [analyze(g.copy(), sc) for g in sel[:40]]
    key = lambda n: np.array([r[n] for r in rows])
    print(f"analysed {len(rows)} grids, K range {int(key('K').min())}..{int(key('K').max())}")
    print(f"  SPF numbers (<=K): mean {key('spf').mean():.1%}  | 4-digit SPF: {key('spf4').mean():.1%}")
    print(f"  chain count cnt[n], 1000..K: median {np.median(key('cnt_med')):.0f}; share with exactly 1 chain {key('cnt1').mean():.1%}")
    print(f"  free cells: median {int(np.median(key('free')))} of 112 | load-bearing cells: median {int(np.median(key('load')))} of 112")
    print(f"  single-cell changes: improve {key('up').mean():.2%} | neutral {key('neu').mean():.2%} | small loss(>-30) {key('small').mean():.2%}")
    print("  per grid [K spf% free load up%]:")
    for r in rows[:12]:
        print(f"    K={r['K']:5d} spf={r['spf']:.1%} free={r['free']:3d} load={r['load']:3d} improve={r['up']:.2%} neutral={r['neu']:.2%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data", int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
