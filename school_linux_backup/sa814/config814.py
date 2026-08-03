"""
Configuration for the sa814 simulated-annealing / parallel-tempering solver.

This module is intentionally platform-agnostic: it only builds a plain
``SAConfig`` dataclass plus an argparse surface. Platform specifics (signal
handling, thread counts, default run roots) live in runtime_win.py /
runtime_linux.py.

Two presets are provided, mirroring the two objectives the original
DEAP-based scripts optimized for:

  - ``preset_score_first()``: maximize the "consecutive score" (largest K
    such that every integer 1..K is formable). This is the "real" 814 score.
  - ``preset_count_first()``: maximize the number of formable values in
    [1000, 10000), a softer secondary objective.

The two presets share every field except the energy weights (``w_score``,
``w_look``, ``w_count``, ``w_heur``) and ``want_count``. See core814.py for
how the energy is computed from these weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Grid geometry (matches the original 814_cpu_*.py IND_ROWS / IND_COLS)
# ---------------------------------------------------------------------------
GRID_ROWS = 8
GRID_COLS = 14
GRID_SIZE = GRID_ROWS * GRID_COLS

# Theoretical maximum consecutive score for BOJ 18789 (814-2).
MAX_SCORE = 8142

# Move operator names, in the fixed order used by core814's move dispatcher.
MOVE_NAMES = ("copy_neighbor", "swap_adjacent", "avoid_repeat", "remap", "repair")

# Acceptance-rule / search-mode choices exposed on the CLI.
ACCEPT_MODES = ("sa", "lahc", "dlas")
SEARCH_MODES = ("pt", "anneal")
RESTART_MODES = ("best", "pbest", "current")


@dataclass
class SAConfig:
    # --- identity / persistence -------------------------------------------------
    run_name: str = "default"
    resume: bool = True          # auto-resume if a matching checkpoint exists
    fresh: bool = False          # force a clean start, ignoring any checkpoint
    seed_file: Optional[str] = None   # extra corpus file (8x14 blocks) to seed from
    seed_from_corpus: bool = True     # if False, ignore data/*.txt + prior run outputs and
                                       # start every replica from a fresh random grid
    seed_grids_file: Optional[str] = None
    # Exclusive seeding: if set, every replica is seeded from the 8x14 blocks in
    # THIS file and nothing else -- overrides both seed_from_corpus and
    # seed_file. data/*.txt and this run's own best.txt/records.txt are never
    # read. Grids are used in FILE ORDER (not re-sorted by score), so the
    # caller has exact control over which grid lands on which replica. This
    # exists because --seed-file alone doesn't work for "start from exactly
    # this grid": without --no-seed it gets merged into and usually outranked
    # by the data/*.txt corpus (up to 7666 points); with --no-seed it's never
    # even read (seeding.build_initial_replicas short-circuits to all-random).
    rng_seed: Optional[int] = None    # seeds driver.py's np.random.default_rng(); None means
                                       # every run is nondeterministic (numpy draws OS entropy),
                                       # which makes two runs' outcomes incomparable. Set this to
                                       # get reproducible A/B comparisons between config changes.

    # --- termination --------------------------------------------------------
    max_seconds: Optional[float] = None
    max_iters: Optional[int] = None
    target_score: Optional[int] = None   # stop early once this score is reached

    # --- parallelism ---------------------------------------------------------
    replicas: int = 12          # number of independent chains / PT ladder rungs
    threads: int = 0            # 0 => auto-detect (runtime layer fills this in)

    # --- search algorithm ------------------------------------------------------
    accept_mode: str = "sa"      # one of ACCEPT_MODES
    search_mode: str = "pt"      # one of SEARCH_MODES ("pt" = parallel tempering,
                                 # "anneal" = independent single-chain annealing)
    lahc_len: int = 64           # history length for "lahc" / "dlas" accept modes

    # --- energy weights: E = -(w_score*score + w_look*look + w_count*count + w_heur*heur)
    w_score: float = 1.0
    w_look: float = 0.0020
    w_count: float = 0.0
    w_heur: float = 0.0
    w_triple: float = 0.005     # penalty per 3-cell same-digit chain reachable via an
                                # 8-directional walk that may bend at each step (see
                                # core814.count_triple_chains) -- since a walk may revisit
                                # cells, only 2 same-digit cells are ever needed to form
                                # any length of repeated-digit number, so a 3rd anywhere
                                # reachable is pure waste. Raised 5x from an earlier 0.001
                                # once the counter was corrected to catch bent (not just
                                # straight-line) triples, since the count now matters more.
                                # Calibrated against a realistic worst case of ~200 triples
                                # a search might actually wander through (not the ~2400 of a
                                # fully degenerate all-one-digit grid, which scores near 0
                                # and is never seriously explored) -- 200 * 0.005 = 1.0, so
                                # even that can only just barely brush a single real score
                                # point, never flip it outright.
    want_count: bool = False     # whether to compute the [count_lo, count_hi) formable count
    look_window: int = 400       # how far past the first failure to keep scanning
    count_lo: int = 1000
    count_hi: int = 10000

    # --- move operator mix (must sum to ~1.0; core814 normalizes defensively) ---
    p_copy_neighbor: float = 0.40  # sets a cell to one of its differing neighbor values
                                   # (uniformly, via reservoir sampling), falling back to a
                                   # blind random digit only if every neighbor already
                                   # matches the cell's own value -- see
                                   # core814.apply_copy_neighbor. An earlier separate
                                   # "set_random" move did the same thing under a different
                                   # name and was merged in here (0.20 + 0.20 -> 0.40),
                                   # since it was just this same idea generalized.
    p_swap_adjacent: float = 0.20  # deranges (permutes with no fixed point, so every
                                   # touched cell's VALUE genuinely changes) the target
                                   # cell + a random k in [1,n] of its neighbors as one
                                   # cluster -- see core814.apply_swap_cluster. k=1
                                   # reproduces the original pairwise-exchange exactly.
    p_avoid_repeat: float = 0.34  # force a cell away from a random subset of its neighbor
                                  # values (or copy a neighbor if they're already all
                                  # distinct) -- see core814.apply_avoid_repeat. Directly
                                  # targets the same waste that w_triple penalizes. Raised
                                  # 10% -> 20% -> 34% after real runs showed dramatically
                                  # faster score climbs from fresh random seeds once this
                                  # move and w_triple were introduced. The main two levers
                                  # to tune going forward are this and p_copy_neighbor.
    p_remap: float = 0.06  # picks k in [2,10] uniformly, then deranges just k digits
                           # (so every one of them genuinely changes to a different
                           # digit) -- see core814.apply_remap. Unifies two former
                           # separate moves: remap_pair (swap exactly 2 digits, always
                           # k=2) and remap_full (relabel all 10 via a random
                           # permutation, k=10) were really the same idea at different
                           # k, so their shares combine here (0.01 + 0.05 -> 0.06).
    p_repair: float = 0.0  # targeted move: computes exactly which single cell/digit
                            # change would let the grid form cur_score+1 (the specific
                            # number that's failing right now), by re-running
                            # is_formable's frontier propagation and capturing where
                            # the walk dies instead of just returning False -- see
                            # core814.apply_repair. A calculation, not a guess, unlike
                            # every other move here. Default 0.0 (off) until an A/B
                            # confirms its accept rate actually beats copy_neighbor's;
                            # enable with --p-repair.
    p_edge_bias: float = 0.5     # probability a local move targets a border cell

    # --- adaptive k-distribution learning (opt-in) ----------------------------
    # avoid_repeat/swap_adjacent/remap each draw a k (how many neighbors, or
    # digits, to touch) uniformly by default. If enabled, k is instead drawn
    # from weights learned from observed accept rates, pooled across all
    # replicas and updated periodically -- see core814.weighted_index_choice
    # and driver._update_k_weights. copy_neighbor is excluded: its k provably
    # doesn't affect the outcome (see apply_copy_neighbor), so there is
    # nothing to learn there. Off by default; enable with --adaptive-k.
    adaptive_k: bool = False
    adaptive_k_update_iters: int = 50_000   # total iters between reweighting passes
    adaptive_k_smoothing: float = 2.0       # Laplace smoothing added to accept/attempt
                                             # before computing a rate, so a k that
                                             # hasn't been tried much yet (or got
                                             # unlucky early) isn't zeroed out
    adaptive_k_decay: float = 0.9           # decay applied to old pooled counts at each
                                             # reweighting, so learning can still adapt
                                             # if the "right" k shifts over a long run

    # --- temperature calibration ---------------------------------------------
    cal_samples: int = 2000
    p_hot: float = 0.5           # target acceptance rate at T0
    p_cold: float = 0.01         # target acceptance rate at T_end

    # --- schedule --------------------------------------------------------------
    cycle_iters: int = 2_000_000     # L_CYCLE: iterations per geometric-cooling cycle
    reheat: float = 0.6              # T <- T0 * reheat on stagnation-triggered reheat
    stagnation_iters: int = 500_000  # iters without a new best before reheating
    restart_from: str = "pbest"      # "best" (falls back to "current" after 3 stagnant
                                      # cycles), "pbest" (each replica restarts from its
                                      # own personal best -- default, see anchor_* below),
                                      # or "current" (never restart, always kick in place)

    # --- personal-best anchoring -----------------------------------------------
    # Real runs showed replicas random-walking far below best_score for the
    # entire run (accept rate flat ~40-45%, replica-pair Hamming distance
    # statistically indistinguishable from independent random grids) --
    # once a catastrophic move collapses a replica's score by thousands,
    # nothing pulls it back toward the frontier it came from. This tracks
    # each replica's OWN best (not the global best, which would collapse
    # all replicas onto one basin -- exactly what stagnation-triggered
    # reheat already does) and snaps a replica back to it once it's drifted
    # anchor_margin points below, instead of only checking at
    # stagnation_iters intervals.
    anchor_enabled: bool = True
    anchor_margin: int = 300    # pbest - look_window(400) is where the `look` energy
                                # term goes fully blind to the replica's own frontier
                                # (no formable values left in its window to see); 300
                                # keeps snapback comfortably inside that horizon.
    anchor_kick: int = 0        # cells randomized on snapback. 0 by default -- measured
                                # empirically (see the stagnation investigation) that any
                                # nonzero kick here is self-defeating near a high score:
                                # a real 8-replica run seeded from a 7666 grid held
                                # rock-steady (anchor_gap=0) for 33s, then a
                                # stagnation-triggered reheat's kick collapsed it to
                                # ~1000-1700, and from then on EVERY anchor snapback
                                # (1104 of them logged) immediately re-collapsed itself,
                                # because kick=3 has the same near-certain chance of
                                # breaking a fragile high-score grid as the collapse that
                                # triggered the snap in the first place (median positive
                                # dE near a 7666 grid is ~6129 -- see preset_score_first's
                                # docstring). anchor_gap never recovered from ~6500 for
                                # the rest of that run. A snapback with kick=0 restores
                                # the pristine known-good grid and lets the normal
                                # accept/reject loop explore from there instead.
    elite_size: int = 6         # top-N distinct grids tracked for resampling below
    elite_resample_iters: int = 0   # iters between reassigning the worst pbest_scores
                                     # replicas a random elite grid ("go with the
                                     # winners"). 0 = off; only enable after measuring
                                     # anchoring alone, since it's a second, compounding
                                     # diversity mechanism
    anchor_grace_iters: int = 200_000   # AGGREGATE iterations (same unit as
                                         # stagnation_iters/iters_since_best, i.e.
                                         # replicas * iters_per_segment * n_segments
                                         # summed across the whole population, NOT
                                         # per replica) of immunity from anchoring
                                         # snapback immediately after a reheat
                                         # restores+kicks a replica. 0 = no grace.
                                         # MUST be < stagnation_iters, or reheat
                                         # re-arms grace faster than it can expire
                                         # and anchoring's snapback becomes
                                         # permanently unreachable (driver.drive()
                                         # clamps to stagnation_iters // 2 and warns
                                         # if this is violated -- confirmed on a
                                         # real 16-replica production run where an
                                         # earlier per-replica-unit bug here froze
                                         # n_snaps immediately after the first
                                         # reheat while anchor_gap grew past 4000
                                         # over 6.7 hours of runtime).
                                         #
                                         # Without grace at all, restart_from='pbest'
                                         # reheat's kick is cancelled almost every
                                         # time: kick randomizes anchor_kick cells,
                                         # which near a high score almost always
                                         # collapses it (median positive dE near a
                                         # 7666 grid is ~6129 -- see anchor_kick's
                                         # docstring), and the very next anchoring
                                         # pass sees "score < pbest - anchor_margin"
                                         # and snaps straight back to pbest before the
                                         # kicked state ever gets a chance to explore.
                                         # pbest itself never falls (monotone), so
                                         # without a grace window the kick gets one
                                         # ~0.25s block to prove itself and then is
                                         # reverted -- observed directly as the
                                         # stagnant_cycles 1->2->3->1 loop on frozen
                                         # servers, where reheat fired every cycle but
                                         # never actually escaped the snapback.

    # --- DLAS trapdoor prevention ------------------------------------------------
    # The DLAS accept rule (core814._anneal_one) is `accept if newE == curE or
    # newE < hmax`, where hmax = max(history). On dlas.hpp's original smooth
    # landscape that's a self-tightening bar: the chain descends, hmax follows
    # it down. On 814-2's cliff (breaking one small number collapses score by
    # thousands) it's a one-way ratchet instead: one catastrophic accept fills
    # history with terrible energies, hmax explodes, and nearly every
    # subsequent proposal satisfies newE < hmax -- an unbiased random walk
    # (confirmed empirically: accept rate sat at a flat 40-45% for the ENTIRE
    # duration of every real run analyzed, never declining as score rose).
    # Phase 1's anchoring recovers from this after the fact; these two fields
    # attack the mechanism that causes it.
    max_worsening: float = 25.0   # hard ceiling on hmax: never more than curE +
                                   # max_worsening above the current energy, no matter
                                   # how bad history has become. 0.0 disables. Blocks
                                   # the catastrophic (score-in-the-thousands) accepts
                                   # outright rather than just recovering from them.
                                   # Known side effect: `remap` (a global digit
                                   # relabeling) will go effectively dead above a few
                                   # hundred score points, since it almost always
                                   # collapses a high-scoring grid outright -- this is
                                   # not a bug to work around, it's max_worsening
                                   # correctly recognizing remap can't safely fire there.
    min_worsening: float = 5.0   # 0.0 disables. Hard floor on hmax: the DLAS/LAHC
                                  # accept bar can never collapse below curE +
                                  # min_worsening, no matter how good history has
                                  # become. Pairs with max_worsening (the ceiling);
                                  # min == max makes this exactly threshold accepting.
                                  # Without this, hmax is a monotone-non-increasing
                                  # ratchet: rejected moves can only pull low history
                                  # slots up to curE, never push high slots down, and
                                  # the only path that lowers a slot (an accepted
                                  # improving move) requires newE < hmax, so hmax can
                                  # never be pushed back above where it already is. At
                                  # a local optimum where every proposal is rejected,
                                  # all lahc_len slots converge to curE within one
                                  # window, hmax collapses to exactly curE, and the
                                  # accept rule degrades to pure greedy (newE <= curE)
                                  # forever -- confirmed as the cause of three real
                                  # servers freezing solid at 5408/5498/5797 within
                                  # 5-6 minutes, stagnant_cycles cycling 1->2->3->1
                                  # forever with best never once updating.
    hist_reset_band: float = 2.0  # every flat history fill (fresh start, resume,
                                   # reheat/anchor restore) sets hist[:] = energy +
                                   # hist_reset_band instead of exactly energy. At
                                   # exactly energy, hmax == curE right after a reset,
                                   # which makes DLAS accept ONLY strict improvements --
                                   # pure greedy, unable to move at all once stuck. A
                                   # small band lets the replica drift slightly (in
                                   # score-point-equivalent units, since w_score=1.0)
                                   # instead of freezing solid.

    # --- exact-state cycle prevention --------------------------------------------
    # Once Phase 1/2 keep a replica anchored near its own frontier instead of
    # randomly walking the whole state space, re-visiting a grid it already
    # tried (and rejected/reverted from) becomes a real, measurable possibility
    # rather than a near-zero-probability event in a 10^112-state space. Each
    # replica keeps a ring buffer of Zobrist hashes (core814.ZOBRIST) of its
    # last cycle_buffer accepted states; a proposed move whose resulting grid
    # matches one of them is rejected outright UNLESS it's actually an
    # improvement over the replica's current energy (aspiration -- a genuinely
    # better state is never wasted even if visited before).
    cycle_buffer: int = 64   # 0 = off. Deliberately much smaller than a classic
                              # tabu list's few-thousand-entry memory: the check is
                              # a linear scan over cycle_buffer entries done EVERY
                              # iteration (a hash set would avoid this, but adds
                              # real complexity for a numba kernel), so this trades
                              # memory depth for per-iteration cost. 64 is short-
                              # term "don't immediately undo what I just tried"
                              # memory, not a full visited-set.

    # --- parallel tempering ----------------------------------------------------
    swap_interval: int = 2000    # iterations between adjacent-replica swap attempts

    # --- checkpointing -----------------------------------------------------
    checkpoint_secs: float = 60.0
    print_secs: float = 5.0

    # --- misc --------------------------------------------------------------
    block_seconds: float = 0.25  # target wall-clock time per njit block (Ctrl-C latency)

    def move_probs(self) -> tuple:
        return (
            self.p_copy_neighbor,
            self.p_swap_adjacent,
            self.p_avoid_repeat,
            self.p_remap,
            self.p_repair,
        )

    def cfg_hash(self) -> str:
        """Stable hash of the fields that affect run semantics (not run_name/resume/etc)."""
        semantic_fields = [
            "accept_mode", "search_mode", "lahc_len",
            "w_score", "w_look", "w_count", "w_heur", "w_triple", "want_count",
            "look_window", "count_lo", "count_hi",
            "p_copy_neighbor", "p_swap_adjacent", "p_avoid_repeat", "p_remap", "p_repair",
            "p_edge_bias", "replicas",
            "adaptive_k", "adaptive_k_update_iters", "adaptive_k_smoothing", "adaptive_k_decay",
            "anchor_enabled", "anchor_margin", "anchor_kick", "elite_size", "elite_resample_iters",
            "anchor_grace_iters",
            "max_worsening", "min_worsening", "hist_reset_band", "cycle_buffer",
        ]
        d = asdict(self)
        payload = json.dumps({k: d[k] for k in semantic_fields}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


def preset_score_first(**overrides) -> SAConfig:
    """Optimize primarily for the real consecutive score (like 814_cpu_score_first.py
    was *intended* to do before falling back to the C++ subprocess).

    accept_mode defaults to "dlas", not "sa". The score is "largest K such that
    1..K are ALL formable": breaking the walk for any single small number
    collapses the score by thousands, no matter how good the rest of the grid
    is. Measured empirically on a real 7666/8142 seed grid, essentially every
    worsening single-cell move is this kind of catastrophic collapse (the 1st
    percentile of positive dE was already ~690; the median ~6129) -- there is
    no "typical small nudge" to calibrate a safe Boltzmann temperature against.
    A plain SA schedule hot enough to be useful for from-scratch search is,
    at the very same temperature, hot enough to immediately destroy a
    near-optimal seed (verified: T=560 dropped a 7666 seed to 99 in one
    block). DLAS/LAHC's late-acceptance rule is self-relative to recent
    history rather than an externally calibrated energy scale, so it adapts
    automatically: it only tolerates zero/negative dE once history is already
    excellent, while still accepting enough exploratory moves from a
    mediocre/random start to make real progress. "sa" remains available via
    --accept sa for experimentation, but is not the safe default here.
    """
    cfg = SAConfig(
        run_name="score_first",
        accept_mode="dlas",
        w_score=1.0,
        w_look=0.0020,   # look_window(400) * w_look = 0.8 < 1.0: can never flip a real score point
        w_count=0.0,
        w_heur=0.0,
        want_count=False,
    )
    return _apply_overrides(cfg, overrides)


def preset_count_first(**overrides) -> SAConfig:
    """Optimize primarily for the formable-count secondary objective, with a small
    nudge toward the real score so it doesn't ignore it entirely.

    Also defaults to "dlas": count is smoother than score (a single mutation
    typically flips only a handful of individual formable/not-formable flags,
    so plain SA is less dangerous here than for score_first), but w_score>0
    still contributes the same catastrophic-collapse risk near a high-scoring
    seed, so the safer default is kept consistent across both presets.
    """
    cfg = SAConfig(
        run_name="count_first",
        accept_mode="dlas",
        w_score=0.05,
        w_look=0.0005,
        w_count=1.0,
        w_heur=0.0,
        want_count=True,
    )
    return _apply_overrides(cfg, overrides)


def _apply_overrides(cfg: SAConfig, overrides: dict) -> SAConfig:
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown SAConfig field: {k}")
        setattr(cfg, k, v)
    return cfg


def build_arg_parser(default_preset: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sa814 simulated-annealing / parallel-tempering solver")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--fresh", action="store_true", default=None)
    p.add_argument("--seed-file", type=str, default=None)
    p.add_argument("--no-seed", action="store_true",
                    help="Ignore data/*.txt and any prior run outputs; start every "
                         "replica from a fresh random grid instead of the existing corpus.")
    p.add_argument("--seed-grids", type=str, default=None, dest="seed_grids_file",
                    help="Seed EVERY replica from the 8x14 blocks in this file only. "
                         "Ignores the data/*.txt corpus and any prior run output "
                         "entirely (unlike --seed-file, which is merged into the "
                         "corpus and usually outranked by it).")
    p.add_argument("--rng-seed", type=int, default=None,
                    help="Seed driver.py's RNG for reproducible runs. Without this, two "
                         "runs with identical flags still diverge (numpy draws OS entropy), "
                         "which makes A/B comparisons meaningless.")

    p.add_argument("--seconds", type=float, default=None, dest="max_seconds")
    p.add_argument("--iters", type=int, default=None, dest="max_iters")
    p.add_argument("--target-score", type=int, default=None)

    p.add_argument("--replicas", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)

    p.add_argument("--accept", type=str, default=None, choices=ACCEPT_MODES, dest="accept_mode")
    p.add_argument("--mode", type=str, default=None, choices=SEARCH_MODES, dest="search_mode")
    p.add_argument("--lahc-len", type=int, default=None)

    p.add_argument("--w-score", type=float, default=None)
    p.add_argument("--w-look", type=float, default=None)
    p.add_argument("--w-count", type=float, default=None)
    p.add_argument("--w-heur", type=float, default=None)
    p.add_argument("--w-triple", type=float, default=None)
    p.add_argument("--look-window", type=int, default=None)
    p.add_argument("--p-repair", type=float, default=None,
                    help="Move-mix share for the targeted repair move (core814."
                         "apply_repair); 0.0 (default) disables it. The only move "
                         "probability with a dedicated flag, since it's meant to be "
                         "A/B tested against the default mix directly.")

    p.add_argument("--checkpoint-secs", type=float, default=None)
    p.add_argument("--swap-interval", type=int, default=None)
    p.add_argument("--reheat", type=float, default=None)
    p.add_argument("--stagnation-iters", type=int, default=None)
    p.add_argument("--cycle-iters", type=int, default=None)
    p.add_argument("--restart-from", type=str, default=None, choices=RESTART_MODES,
                    help="On a stagnation-triggered reheat: 'best' resets every replica to "
                         "best_grid (falling back to kicking its own current state after 3 "
                         "consecutive stagnant reheats, for diversity); 'pbest' (default) "
                         "resets each replica to its OWN best grid; 'current' never resets "
                         "to any stored grid at all, always kicking in place.")

    p.add_argument("--no-anchor", action="store_true",
                    help="Disable personal-best anchoring (see anchor_enabled).")
    p.add_argument("--anchor-margin", type=int, default=None)
    p.add_argument("--anchor-kick", type=int, default=None)
    p.add_argument("--elite-size", type=int, default=None)
    p.add_argument("--elite-resample-iters", type=int, default=None)

    p.add_argument("--max-worsening", type=float, default=None,
                    help="Hard ceiling on how bad DLAS/LAHC's history-derived accept "
                         "bar can get; 0.0 disables. Blocks catastrophic collapse-by-"
                         "thousands accepts outright.")
    p.add_argument("--min-worsening", type=float, default=None,
                    help="Hard floor on how good DLAS/LAHC's history-derived accept "
                         "bar can get; 0.0 disables. Prevents the accept bar from "
                         "collapsing to pure greedy (curE) once history converges at "
                         "a local optimum.")
    p.add_argument("--hist-reset-band", type=float, default=None)
    p.add_argument("--anchor-grace-iters", type=int, default=None,
                    help="Per-replica iterations of immunity from anchoring snapback "
                         "immediately after a reheat restores+kicks that replica; "
                         "0 disables the grace window.")

    p.add_argument("--cycle-buffer", type=int, default=None,
                    help="Ring-buffer size for exact-state cycle prevention; 0 disables.")

    p.add_argument("--adaptive-k", action="store_true", default=None,
                    help="Learn per-k acceptance-rate weights for avoid_repeat/"
                         "swap_adjacent/remap's k-selection from this run's own "
                         "observed data, instead of drawing k uniformly.")
    p.add_argument("--adaptive-k-update-iters", type=int, default=None)
    p.add_argument("--adaptive-k-smoothing", type=float, default=None)
    p.add_argument("--adaptive-k-decay", type=float, default=None)
    return p


def config_from_cli(argv=None, default_preset: str = "score_first") -> SAConfig:
    preset_fn = preset_score_first if default_preset == "score_first" else preset_count_first
    cfg = preset_fn()

    parser = build_arg_parser(default_preset)
    ns = parser.parse_args(argv)

    valid_names = {f.name for f in fields(SAConfig)}
    for name in valid_names:
        if not hasattr(ns, name):
            continue
        val = getattr(ns, name)
        if val is not None:
            setattr(cfg, name, val)

    if ns.no_seed:
        cfg.seed_from_corpus = False

    if ns.no_anchor:
        cfg.anchor_enabled = False

    if ns.fresh:
        cfg.resume = False
        cfg.fresh = True

    # want_count just gates whether evaluate() bothers computing the
    # [count_lo, count_hi) formable count at all -- if the caller set a
    # nonzero --w-count (e.g. to fold it into a score_first run, which
    # defaults want_count off), that weight would silently do nothing
    # without this, since core814.evaluate returns count=0 when want_count
    # is False regardless of w_count.
    if cfg.w_count > 0.0:
        cfg.want_count = True

    return cfg
