"""
Seed tournament orchestrator for sa814.

Runs many independent SA candidates ("individuals") in a population, one
subprocess slice at a time, and picks which individual gets the next slice
using DEAP's selTournamentDCD (dominance + crowding distance over a small
multi-objective vector) plus an NYPC-style bounded relative-score formula for
leaderboard/park/retire decisions. There is NO crossover and NO offspring
generation -- each individual anneals independently forever; a retired
individual is replaced only by a fresh random seed.

Why this exists (see the plan doc / commit message for the full data):
the one historical run that ever beat 5500 (final 6549) was ranked LAST of
four runs at the 500M-iteration checkpoint, and made its final improvement
at 3.68B iterations, after which 11.5B more iterations (76% of its total
runtime) produced zero further improvement. So (a) a single run's marginal
value saturates hard, arguing for spreading iterations across many
independent seeds instead of one long run, and (b) early rank does not
predict final outcome, arguing against aggressive early elimination -- hence
weak selection pressure (tournsize effectively 2, a 25% uniform explore
valve, and a hard 500M-iteration floor before any individual may be retired).

Linux-only for the graceful-shutdown path (SIGTERM -> child finishes its
current ~0.25s block and checkpoints). On Windows, Popen.terminate() is
TerminateProcess with no clean checkpoint, so the periodic 60s checkpoint is
the only safety net there.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint
import core814 as core
import seeding

try:
    from deap.tools import emo
except ImportError:  # pragma: no cover
    emo = None

SA814_DIR = Path(__file__).resolve().parent
FORMAT_VERSION = 1


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class Individual:
    ind_id: str
    run_name: str
    state: str = "active"          # active | parked | archived | failed
    origin: str = "founder:random"
    cfg_flags: list = field(default_factory=list)

    # Refreshed from meta.json + a best.txt rescore after every slice.
    best_score: int = -1
    look: int = 0
    triples: int = 0
    total_iters: int = 0
    elapsed_seconds: float = 0.0
    cfg_hash: Optional[str] = None

    # Orchestrator-maintained. NEVER read from meta.json (see
    # iters_since_best's docstring note in driver.py -- it resets on every
    # reheat regardless of whether anything improved, so it cannot answer
    # "how long since THIS individual last set a record").
    stall_iters: int = 0
    recent_gain: int = 0
    slices_run: int = 0
    times_parked: int = 0
    parked_at_round: int = -1
    consecutive_failures: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Individual":
        return Individual(**d)


@dataclass
class TournamentConfig:
    name: str
    replicas: int = 12
    pop_size: int = 6
    max_pop: int = 8
    selection_mode: str = "tournament"   # "tournament" | "uniform"

    slice_iters: int = 100_000_000
    slice_seconds_cap: float = 2700.0

    park_stall_iters: int = 400_000_000
    retire_stall_iters: int = 2_000_000_000
    min_trial_iters: int = 500_000_000
    elite_keep: int = 2
    min_active: int = 2
    p_explore: float = 0.25

    obj_weight_score: float = 1.0
    obj_weight_look: float = 0.5
    obj_weight_triples: float = 0.2
    obj_weight_gain: float = 0.5

    base_seed: int = 1
    solver_entrypoint: str = "linux_score_first.py"   # test suites on non-Linux dev
                                                        # boxes override this to
                                                        # win_score_first.py; production
                                                        # deploys always use the default

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TournamentConfig":
        return TournamentConfig(**{k: v for k, v in d.items() if k in TournamentConfig.__dataclass_fields__})


@dataclass
class SliceResult:
    ok: bool
    returncode: int = 0
    pre_iters: int = 0
    post_iters: int = 0
    delta_iters: int = 0
    pre_score: int = -1
    post_score: int = -1
    cfg_hash: Optional[str] = None
    stop_reason: str = "unknown"
    wall_seconds: float = 0.0
    reset_detected: bool = False
    crashed: bool = False


# ===========================================================================
# Objective scoring (best.txt rescore) + NYPC relative-score formula
# ===========================================================================

class _RescoreScratch:
    """Reusable evaluate() scratch buffers -- avoids seeding.score_grid's
    per-call 400KB stamp allocation when rescoring a whole population."""

    def __init__(self):
        self.dmask = np.zeros((10, core.ROWS), dtype=np.int64)
        self.stamp = np.full(core.UPPER, -1, dtype=np.int64)
        self.buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
        self.gen = 0

    def rescore(self, grid: np.ndarray):
        self.gen += 1
        core.build_dmask(np.ascontiguousarray(grid), self.dmask)
        score, look, _count = core.evaluate(self.dmask, self.stamp, self.gen, 400, 1000, 10000, False, self.buf)
        triples = int(core.count_triple_chains(grid))
        return int(score), int(look), triples


def parse_grid_from_best_txt(run_dir: Path) -> Optional[np.ndarray]:
    grids = seeding.parse_grids_from_file(run_dir / "best.txt")
    return grids[0] if grids else None


def nypc_relative_scores(values: list, maximize: bool) -> list:
    """NYPC challenge-area formula applied to one objective across a
    population: rel = 1e6 * (1 - 0.5*sqrt((n_lose + 0.5*n_draw) / n_tot)).
    Bounded [5e5, 1e6]: last place still gets half of first place's score,
    and the sqrt makes the penalty concave (steepest right at the top).
    Rank-based, so it's scale-invariant between a 3000-point-era population
    and a 6000-point-era one."""
    n = len(values)
    if n == 0:
        return []
    out = []
    for v in values:
        if maximize:
            n_lose = sum(1 for o in values if o > v)
            n_draw = sum(1 for o in values if o == v) - 1
        else:
            n_lose = sum(1 for o in values if o < v)
            n_draw = sum(1 for o in values if o == v) - 1
        rel = 1_000_000.0 * (1.0 - 0.5 * math.sqrt((n_lose + 0.5 * n_draw) / n))
        out.append(rel)
    return out


def compute_nypc_points(pop: list, tcfg: TournamentConfig) -> dict:
    """Returns {ind_id: nypc_points} -- used for leaderboard/park/retire
    ordering and revive's LRU tiebreak, NOT for slice-selection (that's the
    DCD layer)."""
    if not pop:
        return {}
    scores = [i.best_score for i in pop]
    looks = [i.look for i in pop]
    triples = [i.triples for i in pop]
    gains = [i.recent_gain for i in pop]

    rel_score = nypc_relative_scores(scores, maximize=True)
    rel_look = nypc_relative_scores(looks, maximize=True)
    rel_triples = nypc_relative_scores(triples, maximize=False)
    rel_gain = nypc_relative_scores(gains, maximize=True)

    points = {}
    for i, ind in enumerate(pop):
        points[ind.ind_id] = (
            tcfg.obj_weight_score * rel_score[i]
            + tcfg.obj_weight_look * rel_look[i]
            + tcfg.obj_weight_triples * rel_triples[i]
            + tcfg.obj_weight_gain * rel_gain[i]
        )
    return points


# ===========================================================================
# DCD (dominance + crowding distance) selection -- the DEAP-derived layer
# ===========================================================================

class _FitWrap:
    """Minimal stand-in for a DEAP Individual: just enough for
    fitness.dominates() and emo.assignCrowdingDist() to work, without
    dragging the whole creator/toolbox machinery into the orchestrator."""

    def __init__(self, ind: Individual):
        self.ind = ind
        # weights: score max, look max, triples MIN (negate), recent_gain max
        self.wvalues = (float(ind.best_score), float(ind.look), float(-ind.triples), float(ind.recent_gain))
        self.crowding_dist = 0.0

    def dominates(self, other: "_FitWrap") -> bool:
        not_worse = all(a >= b for a, b in zip(self.wvalues, other.wvalues))
        strictly_better = any(a > b for a, b in zip(self.wvalues, other.wvalues))
        return not_worse and strictly_better


def _assign_crowding_distance(wrapped: list) -> None:
    """Same algorithm as deap.tools.emo.assignCrowdingDist, operating on our
    _FitWrap objects directly (avoids needing creator.FitnessMulti/Individual
    boilerplate just to call one function)."""
    if not wrapped:
        return
    n = len(wrapped)
    for w in wrapped:
        w.crowding_dist = 0.0
    n_obj = len(wrapped[0].wvalues)
    for m in range(n_obj):
        wrapped.sort(key=lambda w: w.wvalues[m])
        vmin = wrapped[0].wvalues[m]
        vmax = wrapped[-1].wvalues[m]
        wrapped[0].crowding_dist = float("inf")
        wrapped[-1].crowding_dist = float("inf")
        if vmax == vmin:
            continue
        for k in range(1, n - 1):
            wrapped[k].crowding_dist += (wrapped[k + 1].wvalues[m] - wrapped[k - 1].wvalues[m]) / (vmax - vmin)


def dcd_better(a: _FitWrap, b: _FitWrap, rng: np.random.Generator) -> _FitWrap:
    """selTournamentDCD's internal tourn(): dominance first, then crowding
    distance (prefer the more isolated / diverse point), then a coin flip.
    Implemented directly rather than calling deap.tools.selTournamentDCD(pop,
    k), which requires k == len(pop) to be divisible by 4 -- a constraint
    our pool sizes (3, 6, 8) routinely violate."""
    if a.dominates(b):
        return a
    if b.dominates(a):
        return b
    if a.crowding_dist < b.crowding_dist:
        return b
    if a.crowding_dist > b.crowding_dist:
        return a
    return a if rng.random() <= 0.5 else b


def sort_nondominated_front0(pop: list) -> list:
    """Returns the ind_ids in the first (best) Pareto front. Used only for
    diagnostics/tests; selection itself uses dcd_better pairwise, matching
    selTournamentDCD's actual behavior rather than a full NSGA-II sort."""
    wrapped = [_FitWrap(i) for i in pop]
    front0 = []
    for w in wrapped:
        if not any(other.dominates(w) for other in wrapped if other is not w):
            front0.append(w.ind.ind_id)
    return front0


# ===========================================================================
# Selection: who gets the next slice
# ===========================================================================

def select_for_slice(individuals: list, tcfg: TournamentConfig, rng: np.random.Generator, round_no: int):
    active = [i for i in individuals if i.state == "active"]
    if not active:
        revive(individuals, tcfg, round_no)
        active = [i for i in individuals if i.state == "active"]
    if not active:
        return None

    if tcfg.selection_mode == "uniform":
        return active[int(rng.integers(len(active)))]

    if rng.random() < tcfg.p_explore:
        return active[int(rng.integers(len(active)))]

    wrapped = {i.ind_id: _FitWrap(i) for i in active}
    _assign_crowding_distance(list(wrapped.values()))
    a, b = rng.choice(active, size=2, replace=len(active) < 2)
    return dcd_better(wrapped[a.ind_id], wrapped[b.ind_id], rng).ind


def revive(individuals: list, tcfg: TournamentConfig, round_no: int) -> None:
    """The user's central requirement: a champion that stalls gets parked
    like anyone else, and parked individuals -- including a former champion
    -- come back. LRU ordering (not score) guarantees every individual is
    eventually revived; stall_iters is reset to 0 on revival, which is what
    makes the revival real rather than nominal (without it, a revived
    individual would just get parked again after one slice)."""
    points = compute_nypc_points([i for i in individuals if i.state in ("parked", "archived")], tcfg)
    parked = [i for i in individuals if i.state == "parked"]
    if not parked:
        parked = [i for i in individuals if i.state == "archived"]
    if not parked:
        return
    parked.sort(key=lambda i: (i.parked_at_round, -points.get(i.ind_id, 0.0)))
    n = max(tcfg.min_active, len(parked) // 3, 1)
    for ind in parked[:n]:
        ind.state = "active"
        ind.stall_iters = 0


# ===========================================================================
# Slice execution
# ===========================================================================

def run_dir_for(ind: Individual) -> Path:
    return checkpoint.run_root(SA814_DIR, ind.run_name)


def run_slice(tcfg: TournamentConfig, ind: Individual, stop_flag: dict) -> SliceResult:
    run_dir = run_dir_for(ind)
    pre_meta = checkpoint.load_meta(run_dir) or {}
    pre_iters = int(pre_meta.get("total_iters", 0))
    pre_score = int(pre_meta.get("best_score", -1))
    pre_elapsed = float(pre_meta.get("elapsed_seconds", 0.0))
    pre_hash = pre_meta.get("cfg_hash")
    first_slice = ind.slices_run == 0

    argv = [tcfg_python(), str(SA814_DIR / tcfg.solver_entrypoint),
            "--run-name", ind.run_name,
            *ind.cfg_flags,
            "--iters", str(pre_iters + tcfg.slice_iters),
            "--seconds", str(pre_elapsed + tcfg.slice_seconds_cap)]
    if first_slice:
        argv.append("--fresh")

    log_path = run_dir / "slice.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    proc = None
    try:
        with open(log_path, "a", encoding="utf-8") as logf:
            popen_kwargs = dict(cwd=str(SA814_DIR), stdout=logf, stderr=subprocess.STDOUT)
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(argv, **popen_kwargs)
            stop_flag["proc"] = proc
            try:
                rc = proc.wait(timeout=tcfg.slice_seconds_cap + 300)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    rc = proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait(timeout=30)
    except Exception as exc:  # noqa: BLE001 -- a crashed child must never kill the tournament
        return SliceResult(ok=False, crashed=True, stop_reason=f"exception: {exc!r}",
                            wall_seconds=time.perf_counter() - t0)
    finally:
        stop_flag["proc"] = None
    wall = time.perf_counter() - t0

    post_meta = checkpoint.load_meta(run_dir)
    if post_meta is None:
        return SliceResult(ok=False, returncode=rc, crashed=True, stop_reason="crash",
                            pre_iters=pre_iters, pre_score=pre_score, wall_seconds=wall)

    post_iters = int(post_meta.get("total_iters", 0))
    post_score = int(post_meta.get("best_score", -1))
    post_hash = post_meta.get("cfg_hash")

    reset_detected = False
    if pre_hash is not None and post_hash != pre_hash:
        reset_detected = True
    if post_iters < pre_iters:
        reset_detected = True

    stop_reason = "unknown"
    try:
        tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-30:]
        for line in reversed(tail):
            if "stopping: reached max_iters" in line:
                stop_reason = "max_iters"; break
            if "stopping: reached max_seconds" in line:
                stop_reason = "max_seconds"; break
            if "stopping: reached target_score" in line:
                stop_reason = "target_score"; break
            if "received signal" in line:
                stop_reason = "signal"; break
    except OSError:
        pass

    return SliceResult(
        ok=True, returncode=rc, pre_iters=pre_iters, post_iters=post_iters,
        delta_iters=max(0, post_iters - pre_iters), pre_score=pre_score, post_score=post_score,
        cfg_hash=post_hash, stop_reason=stop_reason, wall_seconds=wall, reset_detected=reset_detected,
    )


def tcfg_python() -> str:
    return sys.executable


# ===========================================================================
# Applying a slice result to an Individual's bookkeeping
# ===========================================================================

def is_elite(ind: Individual, population: list, tcfg: TournamentConfig) -> bool:
    """Top elite_keep individuals by best_score (ties broken by ind_id for a
    stable ordering) are never archived -- they may still be parked (that's
    exactly the "champion isn't exempt from rotation" requirement), but their
    grid and run history are kept forever rather than being retired."""
    if tcfg.elite_keep <= 0:
        return False
    ranked = sorted(population, key=lambda i: (-i.best_score, i.ind_id))
    return ind.ind_id in {i.ind_id for i in ranked[:tcfg.elite_keep]}


def apply_slice_result(ind: Individual, res: SliceResult, tcfg: TournamentConfig, round_no: int,
                        scratch: _RescoreScratch, population: Optional[list] = None) -> list:
    events = []
    ind.slices_run += 1

    if not res.ok or res.crashed:
        ind.consecutive_failures += 1
        events.append(("slice_failed", ind.ind_id, res.stop_reason))
        if ind.consecutive_failures >= 3:
            ind.state = "failed"
            events.append(("individual_failed", ind.ind_id, ind.consecutive_failures))
        return events

    if res.reset_detected:
        events.append(("destructive_reset", ind.ind_id,
                        f"pre_iters~{res.pre_iters} post_iters={res.post_iters} "
                        f"pre_hash vs post_hash mismatch or total_iters decreased"))
        ind.state = "failed"
        return events

    ind.consecutive_failures = 0
    ind.total_iters = res.post_iters
    ind.cfg_hash = res.cfg_hash

    grid = parse_grid_from_best_txt(run_dir_for(ind))
    if grid is not None:
        score, look, triples = scratch.rescore(grid)
    else:
        score, look, triples = res.post_score, ind.look, ind.triples

    improved = score > ind.best_score
    ind.recent_gain = max(0, score - ind.best_score) if improved else max(0, ind.recent_gain - 1)
    if improved:
        ind.best_score = score
        ind.stall_iters = 0
        events.append(("new_record", ind.ind_id, score))
    else:
        ind.stall_iters += res.delta_iters
    ind.look = look
    ind.triples = triples

    elite = is_elite(ind, population, tcfg) if population is not None else False
    if ind.total_iters >= tcfg.min_trial_iters:
        if ind.stall_iters >= tcfg.retire_stall_iters and not elite:
            ind.state = "archived"
            events.append(("archived", ind.ind_id, ind.stall_iters))
        elif ind.stall_iters >= tcfg.park_stall_iters:
            ind.state = "parked"
            ind.times_parked += 1
            ind.parked_at_round = round_no
            events.append(("parked", ind.ind_id, ind.stall_iters))

    return events


# ===========================================================================
# Population maintenance: elitism + fresh-seed replacement (no crossover)
# ===========================================================================

def maybe_replace_archived(individuals: list, tcfg: TournamentConfig, rng: np.random.Generator,
                            next_seq_fn) -> list:
    """When an individual is archived and the active/parked/failed-but-alive
    population is below pop_size, open a brand new random-seed individual --
    NOT a crossover child. See the plan doc's "honest assessment" for why
    crossover was deliberately excluded from this design."""
    events = []
    alive = [i for i in individuals if i.state in ("active", "parked")]
    if len(alive) >= tcfg.pop_size or len(individuals) >= tcfg.max_pop:
        return events
    seq = next_seq_fn()
    ind_id = f"i{seq:03d}"
    seed = tcfg.base_seed + seq
    new_ind = Individual(
        ind_id=ind_id,
        run_name=f"{tcfg.name}__{ind_id}",
        origin="replacement:random",
        cfg_flags=["--fresh", "--no-seed", "--rng-seed", str(seed), "--replicas", str(tcfg.replicas)],
    )
    individuals.append(new_ind)
    events.append(("new_individual", ind_id, "replacement:random"))
    return events


# ===========================================================================
# State persistence
# ===========================================================================

def tournament_dir(name: str) -> Path:
    return SA814_DIR / "tournament" / name


def state_path(name: str) -> Path:
    return tournament_dir(name) / "state.json"


def save_state(state: dict) -> None:
    d = tournament_dir(state["tournament"]["name"])
    d.mkdir(parents=True, exist_ok=True)
    path = d / "state.json"
    prev = d / "state.prev.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    if path.exists():
        try:
            path.replace(prev)
        except OSError:
            pass
    checkpoint._replace_with_retry(tmp, path)


def load_state(name: str) -> Optional[dict]:
    path = state_path(name)
    if not path.exists():
        prev = tournament_dir(name) / "state.prev.json"
        if prev.exists():
            path = prev
        else:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def log_event(name: str, event_type: str, *args) -> None:
    d = tournament_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "type": event_type, "args": args}
    with open(d / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ===========================================================================
# Founding a fresh tournament
# ===========================================================================

def make_founders(tcfg: TournamentConfig) -> list:
    founders = []
    n_random = max(1, tcfg.pop_size - 1)
    for k in range(n_random):
        ind_id = f"i{k:03d}"
        seed = tcfg.base_seed + k
        founders.append(Individual(
            ind_id=ind_id, run_name=f"{tcfg.name}__{ind_id}", origin="founder:random",
            cfg_flags=["--fresh", "--no-seed", "--rng-seed", str(seed), "--replicas", str(tcfg.replicas)],
        ))
    if len(founders) < tcfg.pop_size:
        k = len(founders)
        ind_id = f"i{k:03d}"
        founders.append(Individual(
            ind_id=ind_id, run_name=f"{tcfg.name}__{ind_id}", origin="founder:corpus",
            cfg_flags=["--fresh", "--rng-seed", str(tcfg.base_seed + k), "--replicas", str(tcfg.replicas)],
        ))
    return founders


# ===========================================================================
# Main loop
# ===========================================================================

def install_signal_handlers(stop_flag: dict) -> None:
    def _handler(signum, frame):
        if stop_flag.get("shutdown"):
            proc = stop_flag.get("proc")
            if proc is not None:
                proc.kill()
            os._exit(1)
        stop_flag["shutdown"] = True
        proc = stop_flag.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _handler)


def run_tournament(tcfg: TournamentConfig, max_rounds: Optional[int] = None) -> dict:
    saved = load_state(tcfg.name)
    if saved is not None:
        tcfg = TournamentConfig.from_dict(saved["tournament"])
        individuals = [Individual.from_dict(d) for d in saved["individuals"]]
        round_no = saved.get("round", 0)
        next_seq = saved.get("next_ind_seq", len(individuals))
    else:
        individuals = make_founders(tcfg)
        round_no = 0
        next_seq = len(individuals)

    rng = np.random.default_rng(tcfg.base_seed + 9999)
    scratch = _RescoreScratch()
    stop_flag = {"shutdown": False, "proc": None}
    install_signal_handlers(stop_flag)

    def next_seq_fn():
        nonlocal next_seq
        v = next_seq
        next_seq += 1
        return v

    def dump_state():
        return {
            "format_version": FORMAT_VERSION,
            "tournament": tcfg.to_dict(),
            "round": round_no,
            "next_ind_seq": next_seq,
            "individuals": [i.to_dict() for i in individuals],
        }

    while not stop_flag["shutdown"]:
        if max_rounds is not None and round_no >= max_rounds:
            break
        round_no += 1

        ind = select_for_slice(individuals, tcfg, rng, round_no)
        if ind is None:
            log_event(tcfg.name, "tournament_exhausted", round_no)
            break

        res = run_slice(tcfg, ind, stop_flag)
        events = apply_slice_result(ind, res, tcfg, round_no, scratch, population=individuals)
        for e in events:
            log_event(tcfg.name, *e)

        events2 = maybe_replace_archived(individuals, tcfg, rng, next_seq_fn)
        for e in events2:
            log_event(tcfg.name, *e)

        save_state(dump_state())

        if stop_flag["shutdown"]:
            break

    return dump_state()


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--max-pop", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=12)
    ap.add_argument("--slice-iters", type=int, default=100_000_000)
    ap.add_argument("--park-stall-iters", type=int, default=400_000_000)
    ap.add_argument("--retire-stall-iters", type=int, default=2_000_000_000)
    ap.add_argument("--min-trial-iters", type=int, default=500_000_000)
    ap.add_argument("--selection-mode", choices=("tournament", "uniform"), default="tournament")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = TournamentConfig(
        name=args.name, pop_size=args.pop_size, max_pop=args.max_pop, replicas=args.replicas,
        slice_iters=args.slice_iters, park_stall_iters=args.park_stall_iters,
        retire_stall_iters=args.retire_stall_iters, min_trial_iters=args.min_trial_iters,
        selection_mode=args.selection_mode, base_seed=args.base_seed,
    )
    final = run_tournament(cfg, max_rounds=args.max_rounds)
    print(json.dumps({"round": final["round"], "n_individuals": len(final["individuals"])}, indent=2))


if __name__ == "__main__":
    main()
