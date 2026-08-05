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
valve, and a hard iteration floor before any individual may be retired).

Founders are ALWAYS random seeds (--no-seed), never corpus-seeded. Real
production data: three separate corpus-seeded individuals (one per server),
each starting from the corpus's 7666-point grid, recorded that score once on
their first slice and then made ZERO further improvement across 148M-305M
combined iterations. A high-scoring starting grid is a dead end here, not a
head start (consistent with anchor_kick's docstring: a 7666-seeded run
collapsed on its first reheat kick and never recovered). Diversity comes
only from independent rng_seed values, never from the corpus.

CRITICAL: cfg_flags must NEVER contain "--fresh". Every slice after the
first must resume from the previous slice's checkpoint -- --fresh forces
driver.py to start over from total_iters=0 every single time (see
run_slice's "first_slice" handling below). This bug shipped once already:
it was in cfg_flags at individual-creation time in an earlier revision,
which made every "slice" a deterministic replay of the same trajectory
capped at ever-larger --iters values, not real progress. Do not reintroduce
it.

Linux-only for the graceful-shutdown path (SIGTERM -> child finishes its
current ~0.25s block and checkpoints) and for the PID lockfile. On Windows,
Popen.terminate() is TerminateProcess with no clean checkpoint, and os.kill
can't be used to probe a PID, so both are skipped there -- this module is
developed/tested on Windows but only ever deployed on Linux servers.
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
FORMAT_VERSION = 2

# A real cfg_hash-mismatch reset (driver.py's destructive-reset branch,
# triggered by changing a semantic_fields entry like --replicas) zeroes
# total_iters outright. An abrupt kill (session/SSH drop, OOM, etc.) can
# instead lose up to one checkpoint_secs (default 60s) interval's worth of
# progress -- observed in production as 1.3M-8.8M-iteration drops (1-4% of
# total), NOT a reset to ~0. Anything smaller than this tolerance is treated
# as a benign regression: log it, resync ind.total_iters to the disk value,
# and count one strike. Anything at or above it (or an actual cfg_hash
# mismatch) is treated as fatal.
RESET_TOLERANCE_ITERS = 20_000_000


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
    stall_iters: int = 0             # resets to 0 on revive() -- gates park
    lifetime_stall_iters: int = 0    # does NOT reset on revive() -- gates
                                      # archive. Without this second counter,
                                      # stall_iters resetting on every revive
                                      # makes retire_stall_iters unreachable
                                      # (an individual can be parked+revived
                                      # forever without its "true" idle time
                                      # ever accumulating past park_stall_iters).
    recent_gain: float = 0.0         # exponentially-decayed sum of per-slice
                                      # score improvements, EXCLUDING the
                                      # first slice (which has no prior
                                      # baseline and would otherwise spike to
                                      # ~best_score itself, drowning out
                                      # look/triples in DCD selection).
    slices_run: int = 0
    times_parked: int = 0
    parked_at_round: int = -1
    consecutive_failures: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Individual":
        valid = {k: v for k, v in d.items() if k in Individual.__dataclass_fields__}
        return Individual(**valid)


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
    min_trial_iters: int = 150_000_000   # was 500M; production slices land
                                          # ~30-55M each (see slice_seconds_cap
                                          # note below), so 500M needed 10-17
                                          # slices to even reach the gate.
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
    post_elapsed: float = 0.0
    cfg_hash: Optional[str] = None
    stop_reason: str = "unknown"
    wall_seconds: float = 0.0
    hash_mismatch: bool = False      # real cfg_hash change -- fatal, no tolerance
    iter_regression: bool = False    # total_iters dropped by >= RESET_TOLERANCE_ITERS -- fatal
    minor_regression: bool = False   # total_iters dropped by < RESET_TOLERANCE_ITERS -- tolerated
    crashed: bool = False

    @property
    def reset_detected(self) -> bool:
        """Fatal-only view, kept for callers that just want "was this
        catastrophic" without caring which of the two fatal reasons fired."""
        return self.hash_mismatch or self.iter_regression


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
    Bounded (5e5, 1e6]: last place still gets roughly half of first place's
    score (exactly half only in the n->infinity limit), and the sqrt makes
    the penalty concave (steepest right at the top). Rank-based, so it's
    scale-invariant between a 3000-point-era population and a 6000-point-era
    one."""
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
    """Returns {ind_id: nypc_points} -- used for leaderboard/park/revive
    ordering, the DCD-degenerate-pool fallback (see select_for_slice), and
    revive's LRU tiebreak. NOT used for normal slice-selection (that's the
    DCD layer, when the active pool is large enough for it to mean anything)."""
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
    boilerplate just to call one function). Mutates the crowding_dist field
    on the given _FitWrap objects; sorts the input list in place (callers
    pass a throwaway list(...) copy for this reason)."""
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

# NOTE on small pools: with few active individuals and 4 objectives,
# assignCrowdingDist marks both the min and max of EVERY objective as
# infinite crowding distance, so most members sit at some objective's
# extreme and dcd_better falls through dominance straight to a coin flip.
# That is intentional, not a bug to work around: dominance is still checked
# first (a clear winner still wins), and degrading to a fair coin flip
# between non-dominated candidates in a tiny pool is exactly what preserves
# the "don't eliminate a currently-behind-but-diverse candidate" property
# DCD selection exists for. An earlier revision of this function replaced
# that coin flip with a deterministic NYPC-scalar comparison for pools under
# 4, which made one individual win nearly every round and starve its
# sibling of slices entirely -- caught by TestV5bParkReviveIntegration and
# TestV6NewSeedReplacement both going from "at least one park/archive event"
# to zero. Do not reintroduce a small-pool fallback without re-verifying
# those two tests.


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

    if len(active) == 1:
        return active[0]

    wrapped = {i.ind_id: _FitWrap(i) for i in active}
    _assign_crowding_distance(list(wrapped.values()))
    ia, ib = rng.integers(len(active)), rng.integers(len(active))
    while ib == ia:
        ib = rng.integers(len(active))
    a, b = active[int(ia)], active[int(ib)]
    return dcd_better(wrapped[a.ind_id], wrapped[b.ind_id], rng).ind


def revive(individuals: list, tcfg: TournamentConfig, round_no: int) -> None:
    """The user's central requirement: a champion that stalls gets parked
    like anyone else, and parked individuals -- including a former champion
    -- come back. LRU ordering (not score) guarantees every individual is
    eventually revived; stall_iters is reset to 0 on revival, which is what
    makes the revival real rather than nominal (without it, a revived
    individual would just get parked again after one slice).

    Falls back through parked -> archived -> failed as each pool empties, so
    a population that has entirely failed (e.g. from a bug, or a run of bad
    luck) is NOT permanently dead -- it gets a chance to resume from
    wherever its checkpoint last stood, rather than the tournament emitting
    tournament_exhausted and stopping outright."""
    candidates = [i for i in individuals if i.state in ("parked", "archived", "failed")]
    points = compute_nypc_points(candidates, tcfg)
    pool = [i for i in individuals if i.state == "parked"]
    if not pool:
        pool = [i for i in individuals if i.state == "archived"]
    if not pool:
        pool = [i for i in individuals if i.state == "failed"]
    if not pool:
        return
    pool.sort(key=lambda i: (i.parked_at_round, -points.get(i.ind_id, 0.0)))
    n = max(tcfg.min_active, len(pool) // 3, 1)
    for ind in pool[:n]:
        ind.state = "active"
        ind.stall_iters = 0
        ind.consecutive_failures = 0


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

    # ind.cfg_flags must never contain "--fresh" (see module docstring) --
    # this is the ONLY place --fresh is ever added, and only for the
    # genuinely first slice of this individual's life. Every later slice
    # omits it, so driver.py resumes from the checkpoint written by the
    # previous slice instead of restarting at total_iters=0.
    assert "--fresh" not in ind.cfg_flags, (
        f"{ind.ind_id}: cfg_flags must not contain --fresh (would force every "
        f"slice to restart from scratch); got {ind.cfg_flags!r}"
    )
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
    post_elapsed = float(post_meta.get("elapsed_seconds", pre_elapsed))

    # A nonzero exit is the child having crashed or been killed abnormally
    # -- a clean stop (max_iters/max_seconds/target_score/signal-triggered
    # graceful shutdown) always exits 0 (driver.py's stop paths all fall
    # through to a normal return). Without this check, a child that dies
    # instantly on a bad flag but leaves a stale meta.json from an earlier
    # slice was silently treated as ok=True, which also reset
    # consecutive_failures to 0 every time -- the 3-strikes guard could
    # never trip for this failure class.
    if rc != 0:
        return SliceResult(ok=False, returncode=rc, crashed=False, stop_reason=f"nonzero_exit_{rc}",
                            pre_iters=pre_iters, pre_score=pre_score, post_iters=post_iters,
                            post_score=post_score, post_elapsed=post_elapsed, wall_seconds=wall)

    hash_mismatch = pre_hash is not None and post_hash != pre_hash
    drop = pre_iters - post_iters
    iter_regression = drop >= RESET_TOLERANCE_ITERS
    minor_regression = (not iter_regression) and drop > 0

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
            if "received signal" in line or "signal received" in line:
                stop_reason = "signal"; break
    except OSError:
        pass

    return SliceResult(
        ok=True, returncode=rc, pre_iters=pre_iters, post_iters=post_iters,
        delta_iters=max(0, post_iters - pre_iters), pre_score=pre_score, post_score=post_score,
        post_elapsed=post_elapsed, cfg_hash=post_hash, stop_reason=stop_reason, wall_seconds=wall,
        hash_mismatch=hash_mismatch, iter_regression=iter_regression, minor_regression=minor_regression,
    )


def tcfg_python() -> str:
    return sys.executable


# ===========================================================================
# Applying a slice result to an Individual's bookkeeping
# ===========================================================================

def is_elite(ind: Individual, population: list, tcfg: TournamentConfig) -> bool:
    """Top elite_keep individuals BY best_score, among live (active/parked)
    individuals only, are never archived -- they may still be parked (that's
    exactly the "champion isn't exempt from rotation" requirement), but their
    grid and run history are kept forever rather than being retired. Dead
    (failed/archived) individuals are excluded from the ranking so they can
    never occupy an elite slot and block a live individual's protection."""
    if tcfg.elite_keep <= 0:
        return False
    live = [i for i in population if i.state in ("active", "parked")]
    ranked = sorted(live, key=lambda i: (-i.best_score, i.ind_id))
    return ind.ind_id in {i.ind_id for i in ranked[:tcfg.elite_keep]}


def _effective_min_trial_iters(ind: Individual, tcfg: TournamentConfig) -> int:
    """Deterministic +/-20% jitter per individual (derived from ind_id, not
    RNG, so it's stable across restarts without needing to persist it)
    prevents the whole population from crossing min_trial_iters in the same
    round and parking as a synchronized cliff."""
    digits = "".join(ch for ch in ind.ind_id if ch.isdigit())
    n = int(digits) if digits else 0
    spread = ((n * 2654435761) % 1000) / 1000.0   # Knuth multiplicative hash, deterministic
    jitter = 0.8 + 0.4 * spread
    return int(tcfg.min_trial_iters * jitter)


def apply_slice_result(ind: Individual, res: SliceResult, tcfg: TournamentConfig, round_no: int,
                        scratch: _RescoreScratch, population: Optional[list] = None) -> list:
    events = []
    ind.slices_run += 1

    if not res.ok or res.crashed:
        ind.consecutive_failures += 1
        events.append(("slice_failed", ind.ind_id, res.stop_reason, res.returncode))
        if ind.consecutive_failures >= 3:
            ind.state = "failed"
            events.append(("individual_failed", ind.ind_id, ind.consecutive_failures))
        return events

    if res.hash_mismatch:
        events.append(("destructive_reset_hash", ind.ind_id,
                        f"cfg_hash changed unexpectedly (pre != post) at total_iters~{res.pre_iters}"))
        ind.state = "failed"
        return events

    if res.iter_regression:
        events.append(("destructive_reset_iters", ind.ind_id,
                        f"pre_iters={res.pre_iters} post_iters={res.post_iters} "
                        f"(dropped {res.pre_iters - res.post_iters:,}, >= tolerance {RESET_TOLERANCE_ITERS:,})"))
        ind.state = "failed"
        return events

    if res.minor_regression:
        ind.consecutive_failures += 1
        events.append(("minor_iter_regression", ind.ind_id,
                        f"pre_iters={res.pre_iters} post_iters={res.post_iters} "
                        f"(dropped {res.pre_iters - res.post_iters:,}, tolerated; resyncing to disk value)"))
        if ind.consecutive_failures >= 3:
            ind.state = "failed"
            events.append(("individual_failed", ind.ind_id, ind.consecutive_failures))
            return events
        # Fall through -- don't return early. The rest of this function
        # re-syncs ind.total_iters/best_score/etc to the (lower, but true)
        # disk values below, exactly as a normal successful slice would.
    else:
        ind.consecutive_failures = 0

    ind.total_iters = res.post_iters
    ind.elapsed_seconds = res.post_elapsed
    ind.cfg_hash = res.cfg_hash

    grid = parse_grid_from_best_txt(run_dir_for(ind))
    if grid is not None:
        score, look, triples = scratch.rescore(grid)
    else:
        score, look, triples = res.post_score, ind.look, ind.triples

    prev_best = ind.best_score
    improved = score > prev_best
    is_first_slice = ind.slices_run == 1
    if is_first_slice:
        # No prior baseline to measure momentum against -- discovering the
        # founder grid's own score isn't evidence of ongoing improvement.
        gain_this_slice = 0.0
    else:
        gain_this_slice = float(max(0, score - prev_best))
    ind.recent_gain = ind.recent_gain * 0.6 + gain_this_slice

    if improved:
        ind.best_score = score
        ind.stall_iters = 0
        ind.lifetime_stall_iters = 0
        events.append(("new_record", ind.ind_id, score))
    else:
        ind.stall_iters += res.delta_iters
        ind.lifetime_stall_iters += res.delta_iters
    ind.look = look
    ind.triples = triples

    pop_for_elite = population if population is not None else [ind]
    elite = is_elite(ind, pop_for_elite, tcfg)
    min_trial = _effective_min_trial_iters(ind, tcfg)
    if ind.total_iters >= min_trial:
        # lifetime_stall_iters (never reset by revive) gates archive, so a
        # repeatedly-parked-and-revived individual's idle time still
        # accumulates toward retirement. stall_iters (reset by revive) gates
        # park, so a just-revived individual isn't immediately re-parked.
        if ind.lifetime_stall_iters >= tcfg.retire_stall_iters and not elite:
            ind.state = "archived"
            events.append(("archived", ind.ind_id, ind.lifetime_stall_iters))
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
    """When the live (active/parked) population is below pop_size -- which
    happens after an individual is archived, OR after one fails (failed
    individuals are not "alive" either, though revive() can bring them back
    first if the failure pool is checked before this runs out of budget) --
    open a brand new random-seed individual. NOT a crossover child: see the
    plan doc's "honest assessment" for why crossover was deliberately
    excluded, and this module's docstring for why corpus seeding was too
    (production data: 0 improvements from 3 independent 7666-seeded
    individuals across 636M combined iterations)."""
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
        cfg_flags=["--no-seed", "--rng-seed", str(seed), "--replicas", str(tcfg.replicas)],
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


def _config_diff(old: dict, new: dict) -> list:
    keys = TournamentConfig.__dataclass_fields__.keys()
    return [(k, old.get(k), new.get(k)) for k in keys if k in old and k in new and old[k] != new[k]]


# ===========================================================================
# Founding a fresh tournament
# ===========================================================================

def make_founders(tcfg: TournamentConfig) -> list:
    """All founders are independent random seeds. See the module docstring
    for the production evidence that corpus seeding is a dead end here."""
    founders = []
    for k in range(tcfg.pop_size):
        ind_id = f"i{k:03d}"
        seed = tcfg.base_seed + k
        founders.append(Individual(
            ind_id=ind_id, run_name=f"{tcfg.name}__{ind_id}", origin="founder:random",
            cfg_flags=["--no-seed", "--rng-seed", str(seed), "--replicas", str(tcfg.replicas)],
        ))
    return founders


# ===========================================================================
# Single-instance lock (POSIX only -- see module docstring)
# ===========================================================================

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just owned by someone else
    except OSError:
        return False


def acquire_lock(name: str) -> Optional[Path]:
    """Returns the lock path (to be released with release_lock) or None if
    locking is not enforced on this platform. Raises RuntimeError if another
    live orchestrator already holds the lock for this tournament name.

    Without this, two orchestrators started against the same --name pick
    (often identical, since both seed their RNG from the same base_seed+9999)
    individuals each round and launch two solver subprocesses against the
    SAME run_dir/checkpoint.npz.tmp at once -- checkpoint.py has no per-writer
    temp-file uniqueness, so concurrent writers can genuinely corrupt a
    checkpoint or silently interleave meta.json (from one writer) with a
    .npz (from the other), which looks identical to the total_iters
    regression this module already has to tolerate for benign reasons."""
    if os.name != "posix":
        return None
    d = tournament_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    lock_path = d / "orchestrator.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        return lock_path
    except FileExistsError:
        try:
            existing_pid = int(lock_path.read_text(encoding="ascii").strip())
        except (ValueError, OSError):
            existing_pid = None
        if existing_pid is not None and _pid_alive(existing_pid):
            raise RuntimeError(
                f"Tournament '{name}' already has a live orchestrator (pid {existing_pid}, "
                f"lockfile {lock_path}). Refusing to start a second one against the same "
                f"run directories."
            )
        # Stale lock from a dead process -- reclaim it.
        lock_path.unlink(missing_ok=True)
        return acquire_lock(name)


def release_lock(lock_path: Optional[Path]) -> None:
    if lock_path is not None:
        lock_path.unlink(missing_ok=True)


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
        if sig is None:
            continue
        # Respect an already-ignored SIGHUP (that's exactly what `nohup`
        # sets before exec'ing us) -- overwriting it here would silently
        # defeat the one thing nohup exists to do, which is exactly the
        # invocation this module's own docstring and linux_tournament.py
        # recommend ("nohup python3 linux_tournament.py ... &").
        if name == "SIGHUP" and signal.getsignal(sig) == signal.SIG_IGN:
            continue
        signal.signal(sig, _handler)


def run_tournament(tcfg: TournamentConfig, max_rounds: Optional[int] = None,
                    force_config: bool = False) -> dict:
    lock_path = acquire_lock(tcfg.name)
    try:
        return _run_tournament_locked(tcfg, max_rounds, force_config)
    finally:
        release_lock(lock_path)


def _run_tournament_locked(tcfg: TournamentConfig, max_rounds: Optional[int],
                            force_config: bool) -> dict:
    saved = load_state(tcfg.name)
    if saved is not None:
        saved_tcfg = TournamentConfig.from_dict(saved["tournament"])
        diffs = _config_diff(saved_tcfg.to_dict(), tcfg.to_dict())
        if diffs:
            if force_config:
                for k, old, new in diffs:
                    log_event(tcfg.name, "config_override", k, old, new)
            else:
                log_event(tcfg.name, "config_ignored", [[k, old, new] for k, old, new in diffs])
                tcfg = saved_tcfg
        else:
            tcfg = saved_tcfg
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
            save_state(dump_state())
            break

        res = run_slice(tcfg, ind, stop_flag)
        log_event(tcfg.name, "slice_done", ind.ind_id, res.pre_iters, res.post_iters,
                   res.delta_iters, res.stop_reason, round(res.wall_seconds, 1),
                   res.returncode, res.post_score > res.pre_score if res.ok else None)

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
    ap.add_argument("--slice-seconds-cap", type=float, default=2700.0)
    ap.add_argument("--park-stall-iters", type=int, default=400_000_000)
    ap.add_argument("--retire-stall-iters", type=int, default=2_000_000_000)
    ap.add_argument("--min-trial-iters", type=int, default=150_000_000)
    ap.add_argument("--elite-keep", type=int, default=2)
    ap.add_argument("--min-active", type=int, default=2)
    ap.add_argument("--p-explore", type=float, default=0.25)
    ap.add_argument("--obj-weight-score", type=float, default=1.0)
    ap.add_argument("--obj-weight-look", type=float, default=0.5)
    ap.add_argument("--obj-weight-triples", type=float, default=0.2)
    ap.add_argument("--obj-weight-gain", type=float, default=0.5)
    ap.add_argument("--selection-mode", choices=("tournament", "uniform"), default="tournament")
    ap.add_argument("--base-seed", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--force-config", action="store_true",
                     help="If a saved tournament state exists with different config values, "
                          "overwrite them with these CLI values instead of silently keeping "
                          "the saved ones (the default -- config is normally frozen at "
                          "founding time to avoid tripping driver.py's destructive cfg_hash "
                          "reset on any individual whose in-flight cfg_flags would change).")
    args = ap.parse_args(argv)

    cfg = TournamentConfig(
        name=args.name, pop_size=args.pop_size, max_pop=args.max_pop, replicas=args.replicas,
        slice_iters=args.slice_iters, slice_seconds_cap=args.slice_seconds_cap,
        park_stall_iters=args.park_stall_iters, retire_stall_iters=args.retire_stall_iters,
        min_trial_iters=args.min_trial_iters, elite_keep=args.elite_keep, min_active=args.min_active,
        p_explore=args.p_explore, obj_weight_score=args.obj_weight_score, obj_weight_look=args.obj_weight_look,
        obj_weight_triples=args.obj_weight_triples, obj_weight_gain=args.obj_weight_gain,
        selection_mode=args.selection_mode, base_seed=args.base_seed,
    )
    final = run_tournament(cfg, max_rounds=args.max_rounds, force_config=args.force_config)
    leaderboard = sorted(final["individuals"], key=lambda i: -i["best_score"])[:5]
    print(json.dumps({
        "round": final["round"],
        "n_individuals": len(final["individuals"]),
        "top5": [(i["ind_id"], i["state"], i["best_score"], i["total_iters"]) for i in leaderboard],
    }, indent=2))


if __name__ == "__main__":
    main()
