"""
Polish a good grid beyond 9999: low-temperature annealing + extend/reinforce moves with the 5-digit engine.

    python polish.py --start out/best_K9169_B3_seed01.txt --tmax 12000 --seconds 1800 --workers 7 --out polish_out

Each worker writes polish_out/w<id>_best.txt every time ITS best K improves (8 lines x 14 digits) and a
progress line every 60 s.  K is capped at tmax -- if a worker reaches K == tmax, rerun with a larger --tmax.
Nothing from the old corpus is used; the start grid is one WE found.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def read_grid(path):
    rows = [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(rows) == 8 and all(len(r) == 14 and r.isdigit() for r in rows), "need an 8x14 digit grid"
    return np.array([[int(c) for c in r] for r in rows], dtype=np.int8)


def worker(job):
    wid, a = job
    import moves5 as mv
    s = mv.Searcher(read_grid(a.start), seed=1000 + wid, temp=a.temp0, tmax=a.tmax, a=a.a, core_mult=a.core_mult, b5=a.b5)
    t0 = time.time(); last = 0.0
    best = s.best_K
    cyc = ["extend"] * 4 + ["reinforce"] * 4
    out = Path(a.out)
    while time.time() - t0 < a.seconds:
        frac = (time.time() - t0) / a.seconds
        s.temp = a.temp0 * (a.temp1 / a.temp0) ** frac
        s.rand_chunk(2000)
        for name in cyc:
            getattr(s, name)()
        if s.best_K > best:
            best = s.best_K
            g = s.best_grid.reshape(8, 14)
            (out / f"w{wid}_best.txt").write_text("\n".join("".join(str(int(v)) for v in r) for r in g) + "\n")
            (out / f"w{wid}_best.json").write_text(json.dumps({"K": int(best), "t": round(time.time() - t0), "tmax": a.tmax}))
        if time.time() - last > 60:
            last = time.time()
            print(f"{time.strftime('%H:%M:%S')} w{wid}: t={last-t0:5.0f}s best K={best} cur K={s.e.K} E={s.e.H:.1f} holes={s.e.holes} temp={s.temp:.2f}", flush=True)
    return wid, int(best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--tmax", type=int, default=12000)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--temp0", type=float, default=1.5)
    ap.add_argument("--temp1", type=float, default=0.3)
    ap.add_argument("--a", type=float, default=0.0, help="extra weight for small-number holes: 1 + a*(1-n/9999)")
    ap.add_argument("--core-mult", type=float, default=1.0, help="multiplier for holes n <= 9999 (use >>1 when tmax > 9999)")
    ap.add_argument("--b5", type=float, default=0.0, help="5-digit holes: weight 1 + b5*(1-(n-9999)/(tmax-9999))")
    ap.add_argument("--out", default="polish_out")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    with mp.Pool(a.workers) as p:
        res = p.map(worker, [(i, a) for i in range(a.workers)])
    print("FINAL:", sorted(res, key=lambda r: -r[1]))
    b = max(res, key=lambda r: r[1])
    print(f"best K = {b[1]} (worker {b[0]}); grid in {a.out}/w{b[0]}_best.txt" + ("  [reached tmax: raise --tmax]" if b[1] >= a.tmax else ""))
