"""
Long-running island search for 814-2.   (PLAN P4)

  python run.py --state state --out out --workers 7 --zero 5 --random 3 \
                --seed-file out/best_K9888_B1a_seed03.txt --seed-file out/best_K9169_B3_seed01.txt

Islands (independent lineages, never mixed):
  zero    start from the all-zero grid                      (stage 1)
  random  start from a random grid (seeded)                  (stage 1)
  seed    start from a grid WE found earlier (file)          (stage 2 directly)
Stage 1 : numbers 1..9999, plain hole-count energy, fast engine + B3 moves (extend/reinforce).  Cooling cycles; every new cycle re-heats from the
          island's best grid; no improvement for `stag_limit` cycles -> the island is restarted (stagnation escape).
Stage 2 : numbers 1..tmax2 (5-digit), weighted energy (4-digit holes protected).  An island moves 1 -> 2 when its best
          grid has <= advance_holes unreadable numbers in 1..9999.
Dedupe  : image code = blake2b of the lexicographically smallest of the 4 symmetric variants (flip LR/UD/180deg);
          two islands whose best grids share a code are duplicates -> the younger one is restarted.
Safety  : state/state.json written atomically every round (+ .prev); resume is automatic (use --fresh to start over);
          every new global best is re-scored with core814 (independent of the engine) before out/best.txt is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scoring"))

NC = 112
STAGE1_SCHEDS = [(3.0, 0.3), (6.0, 0.6), (4.5, 0.45)]     # first-cycle cooling variety (islands cycle through these)


# --------------------------------------------------------------------------- helpers
def canon_code(flat):
    g = np.asarray(flat, dtype=np.uint8).reshape(8, 14)
    vs = [g, g[:, ::-1], g[::-1, :], g[::-1, ::-1]]
    return hashlib.blake2b(min(np.ascontiguousarray(v).tobytes() for v in vs), digest_size=16).hexdigest()


def read_grid_file(path):
    rows = [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert len(rows) == 8 and all(len(r) == 14 and r.isdigit() for r in rows), f"{path}: need an 8x14 digit grid"
    return [int(c) for r in rows for c in r]


def grid_text(flat):
    g = np.asarray(flat).reshape(8, 14)
    return "\n".join("".join(str(int(v)) for v in r) for r in g) + "\n"


def core_score(flat):
    import core814 as core
    g = np.ascontiguousarray(np.asarray(flat, dtype=np.uint8).reshape(8, 14))
    dm = np.zeros((10, 8), dtype=np.int64)
    st = np.full(core.UPPER, -1, dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    core.build_dmask(g, dm)
    return int(core.evaluate(dm, st, 1, 400, 1000, 10000, False, buf)[0])


def engine_K(flat, tmax):
    import engine5 as eg
    e = eg.Engine(np.array(flat, dtype=np.int8), tmax=tmax)
    return e.K, e.holes


def new_island(iid, kind, rng_seed, cfg, seed_file=None):
    rng = np.random.default_rng(rng_seed)
    if kind == "zero":
        grid = [0] * NC
    elif kind == "random":
        grid = [int(v) for v in rng.integers(0, 10, size=NC)]
    else:
        grid = read_grid_file(seed_file)
    stage = 2 if kind == "seed" else 1
    tmax = cfg["tmax2"] if stage == 2 else 9999
    K, _ = engine_K(grid, tmax)
    return dict(id=iid, kind=kind, seed_file=seed_file, rng_seed=int(rng_seed), stage=stage, grid=grid,
                best_K=int(K), best_grid=list(grid), cycle_elapsed=0.0, cycle_no=0, cycle_best_at_start=int(K),
                stag=0, evals=0, restarts=0, born=time.time(), cur_K=int(K), cur_H=0.0, need_restart=False,
                chunk_no=0)


def stage_params(isl, cfg):
    if isl["stage"] == 1:
        if isl["cycle_no"] == 0:
            t0, t1 = STAGE1_SCHEDS[isl["rng_seed"] % len(STAGE1_SCHEDS)]
        else:
            t0, t1 = cfg["reheat1"], 0.3
        return dict(tmax=9999, a=0.0, cm=1.0, b5=0.0, t0=t0, t1=t1, cycle_s=cfg["cycle1"], chunk=2000)
    t0 = cfg["t2_0"]
    return dict(tmax=cfg["tmax2"], a=3.0, cm=1000.0, b5=3.0, t0=t0, t1=cfg["t2_1"], cycle_s=cfg["cycle2"], chunk=2000)


# --------------------------------------------------------------------------- worker
def run_slice(args):
    isl, slice_s, cfg = args
    import moves5 as mv
    t_start = time.time()
    events = []
    P = stage_params(isl, cfg)
    # B3 recipe from the A/B (best median K): a fast jitted random-Metropolis chunk, then structured chain moves
    # (extend = fix the first hole through a one-cell overlap, reinforce = add a chain to a single-point-of-failure number)
    s = mv.Searcher(np.array(isl["grid"], dtype=np.int8), seed=isl["rng_seed"] * 1_000_003 + isl["chunk_no"],
                    temp=P["t0"], tmax=P["tmax"], a=P["a"], core_mult=P["cm"], b5=P["b5"])
    cyc = ["extend"] * 4 + ["reinforce"] * 4
    improved_this_slice = False
    while time.time() - t_start < slice_s and not isl["need_restart"]:
        frac = min(isl["cycle_elapsed"] / P["cycle_s"], 1.0)
        s.temp = P["t0"] * (P["t1"] / P["t0"]) ** frac
        t1 = time.time()
        s.rand_chunk(P["chunk"])
        for name in cyc:
            getattr(s, name)()
        isl["chunk_no"] += 1
        isl["evals"] += P["chunk"] + len(cyc)
        isl["cycle_elapsed"] += time.time() - t1
        if s.best_K > isl["best_K"]:
            isl["best_K"] = int(s.best_K)
            isl["best_grid"] = [int(v) for v in s.best_grid]
            improved_this_slice = True
            events.append(("new_best", isl["id"], int(s.best_K), isl["stage"]))
        if isl["cycle_elapsed"] >= P["cycle_s"]:
            improved = isl["best_K"] > isl["cycle_best_at_start"]
            isl["stag"] = 0 if improved else isl["stag"] + 1
            events.append(("cycle_end", isl["id"], isl["cycle_no"], isl["best_K"], isl["stag"]))
            isl["cycle_no"] += 1
            isl["cycle_elapsed"] = 0.0
            isl["cycle_best_at_start"] = isl["best_K"]
            if isl["stag"] >= cfg["stag_limit"]:
                isl["need_restart"] = True
                break
            P = stage_params(isl, cfg)
            s.e.set_grid(np.array(isl["best_grid"], dtype=np.int8))      # re-heat from the island's best grid
    isl["grid"] = [int(v) for v in s.e.grid]
    isl["cur_K"] = int(s.e.K)
    isl["cur_H"] = float(s.e.H)
    # stage 1 -> 2 when the best grid is (almost) complete below 10000
    if isl["stage"] == 1 and improved_this_slice and not isl["need_restart"]:
        _K, holes = engine_K(isl["best_grid"], 9999)
        if holes <= cfg["advance_holes"]:
            isl.update(stage=2, grid=list(isl["best_grid"]), cycle_elapsed=0.0, cycle_no=0, stag=0,
                       cycle_best_at_start=isl["best_K"])
            events.append(("stage2", isl["id"], isl["best_K"], holes))
    return isl, events


# --------------------------------------------------------------------------- master
def _init_worker():
    """Workers must NOT inherit the master's handlers (they are forked after those are installed): Ctrl-C is handled by
    the master only, and SIGTERM must simply stop a worker."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)


def atomic_write(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)


def load_state(path: Path):
    for p in (path, path.with_name(path.name + ".prev")):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def log_event(out: Path, ev):
    with open(out / "events.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {json.dumps(ev)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--zero", type=int, default=5)
    ap.add_argument("--random", type=int, default=3)
    ap.add_argument("--seed-file", action="append", default=[])
    ap.add_argument("--base-seed", type=int, default=20260101)
    ap.add_argument("--slice", type=float, default=60.0, help="seconds each island runs per round")
    ap.add_argument("--cycle1", type=float, default=900.0)
    ap.add_argument("--cycle2", type=float, default=1800.0)
    ap.add_argument("--reheat1", type=float, default=1.5)
    ap.add_argument("--t2-0", type=float, default=0.5)
    ap.add_argument("--t2-1", type=float, default=0.2)
    ap.add_argument("--tmax2", type=int, default=20000)
    ap.add_argument("--advance-holes", type=int, default=2)
    ap.add_argument("--stag-limit", type=int, default=4)
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = run forever")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    cfg = dict(slice=a.slice, cycle1=a.cycle1, cycle2=a.cycle2, reheat1=a.reheat1, t2_0=a.t2_0, t2_1=a.t2_1,
               tmax2=a.tmax2, advance_holes=a.advance_holes, stag_limit=a.stag_limit)
    sdir, out = Path(a.state), Path(a.out)
    sdir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    spath = sdir / "state.json"

    st = None if a.fresh else load_state(spath)
    if st:
        print(f"resuming: round {st['round']}, elapsed {st['elapsed']/3600:.2f}h, {len(st['islands'])} islands", flush=True)
    else:
        islands, k = [], 0
        for _ in range(a.zero):
            islands.append(new_island(f"z{k}", "zero", a.base_seed + k, cfg)); k += 1
        for _ in range(a.random):
            islands.append(new_island(f"r{k}", "random", a.base_seed + k, cfg)); k += 1
        for f in a.seed_file:
            islands.append(new_island(f"s{k}", "seed", a.base_seed + k, cfg, seed_file=f)); k += 1
        st = dict(version=1, round=0, elapsed=0.0, next_seed=a.base_seed + k, islands=islands, restarts=0,
                  global_best=dict(K=0, grid=[0] * NC, island=""), seen_codes=[])
        log_event(out, ("start", [(i["id"], i["kind"], i["stage"], i["best_K"]) for i in islands]))
    stop = {"flag": False}

    def _sig(signum, frame):
        if stop["flag"]:
            os._exit(1)
        stop["flag"] = True
        print("\nsignal: finishing the round, saving state ...", flush=True)

    for s in ("SIGINT", "SIGTERM", "SIGHUP"):
        sg = getattr(signal, s, None)
        if sg is None:
            continue
        if s == "SIGHUP" and signal.getsignal(sg) == signal.SIG_IGN:
            continue                                   # respect nohup
        signal.signal(sg, _sig)

    pool = mp.Pool(a.workers, initializer=_init_worker)
    t_last = time.time()
    seen = set(st["seen_codes"])
    try:
        while not stop["flag"]:
            tasks = [(i, cfg["slice"], cfg) for i in st["islands"]]
            results = pool.map(run_slice, tasks, chunksize=1)
            st["islands"] = [r[0] for r in results]
            now = time.time(); st["elapsed"] += now - t_last; t_last = now
            st["round"] += 1
            for isl, evs in results:
                for ev in evs:
                    log_event(out, ev)
                    if ev[0] == "new_best" and ev[2] >= 8000:
                        c = canon_code(isl["best_grid"])
                        if c not in seen:
                            seen.add(c)
                            with open(out / "elites.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps(dict(K=ev[2], island=isl["id"], code=c, grid=isl["best_grid"])) + "\n")
            # global best, re-scored by core814 (independent of the engine)
            top = max(st["islands"], key=lambda i: i["best_K"])
            if top["best_K"] > st["global_best"]["K"]:
                ck = core_score(top["best_grid"])
                if ck > st["global_best"]["K"]:
                    st["global_best"] = dict(K=ck, grid=list(top["best_grid"]), island=top["id"])
                    atomic_write(out / "best.txt", grid_text(top["best_grid"]))
                    with open(out / "records.txt", "a", encoding="utf-8") as f:
                        f.write(f"# K={ck} island={top['id']} round={st['round']} t={st['elapsed']:.0f}s engineK={top['best_K']}\n{grid_text(top['best_grid'])}\n")
                    log_event(out, ("GLOBAL_BEST", ck, top["id"], top["best_K"]))
            # duplicate lineages -> restart the younger island
            codes = {}
            for isl in sorted(st["islands"], key=lambda i: i["born"]):
                if isl["best_K"] < 2000:
                    continue
                c = canon_code(isl["best_grid"])
                if c in codes:
                    isl["need_restart"] = True
                    log_event(out, ("duplicate_lineage", isl["id"], codes[c], isl["best_K"]))
                else:
                    codes[c] = isl["id"]
            # restarts (stagnation / duplicate)
            for idx, isl in enumerate(st["islands"]):
                if isl["need_restart"]:
                    with open(out / "retired.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(id=isl["id"], kind=isl["kind"], K=isl["best_K"], stage=isl["stage"],
                                                evals=isl["evals"], grid=isl["best_grid"])) + "\n")
                    new = new_island(isl["id"], isl["kind"], st["next_seed"], cfg, seed_file=isl["seed_file"])
                    new["restarts"] = isl["restarts"] + 1
                    st["next_seed"] += 1
                    st["restarts"] += 1
                    st["islands"][idx] = new
                    log_event(out, ("restart", isl["id"], isl["best_K"], new["rng_seed"]))
            st["seen_codes"] = sorted(seen)
            atomic_write(spath, json.dumps(st))
            # status line + json
            isl_txt = " ".join(f"{i['id']}:{i['best_K']}{'*' if i['stage'] == 2 else ''}" for i in st["islands"])
            evals = sum(i["evals"] for i in st["islands"])
            line = (f"[814] best={st['global_best']['K']} ({st['global_best']['island']}) | round {st['round']} | "
                    f"{st['elapsed']/3600:.2f}h | evals {evals/1e9:.2f}e9 | restarts {st['restarts']} | {isl_txt}   (*=stage2)")
            print(line, flush=True)
            atomic_write(sdir / "status.json", json.dumps(dict(line=line, global_best=st["global_best"]["K"],
                         islands=[{k: i[k] for k in ("id", "kind", "stage", "best_K", "cur_K", "stag", "cycle_no", "evals", "restarts")}
                                  for i in st["islands"]], round=st["round"], elapsed=st["elapsed"])))
            if a.max_rounds and st["round"] >= a.max_rounds:
                break
    finally:
        pool.terminate()
        atomic_write(spath, json.dumps(st))
        print("state saved:", spath, flush=True)


if __name__ == "__main__":
    main()
