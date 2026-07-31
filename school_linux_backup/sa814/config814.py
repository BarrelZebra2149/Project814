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
MOVE_NAMES = ("set_random", "copy_neighbor", "swap_adjacent", "avoid_repeat", "remap_pair", "remap_full")

# Acceptance-rule / search-mode choices exposed on the CLI.
ACCEPT_MODES = ("sa", "lahc", "dlas")
SEARCH_MODES = ("pt", "anneal")


@dataclass
class SAConfig:
    # --- identity / persistence -------------------------------------------------
    run_name: str = "default"
    resume: bool = True          # auto-resume if a matching checkpoint exists
    fresh: bool = False          # force a clean start, ignoring any checkpoint
    seed_file: Optional[str] = None   # extra corpus file (8x14 blocks) to seed from
    seed_from_corpus: bool = True     # if False, ignore data/*.txt + prior run outputs and
                                       # start every replica from a fresh random grid

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
    p_set_random: float = 0.20   # sets a cell to one of its differing neighbor values
                                 # (uniformly, via reservoir sampling), falling back to a
                                 # blind random digit only if every neighbor already matches
                                 # the cell's own value -- see core814.apply_set_random.
                                 # Adopts avoid_repeat's neighbor-awareness; effectively an
                                 # upgraded copy_neighbor (samples over ALL differing
                                 # neighbors, not just one fixed random direction).
    p_copy_neighbor: float = 0.20
    p_swap_adjacent: float = 0.20
    p_avoid_repeat: float = 0.34  # force a cell away from a random subset of its neighbor
                                  # values (or copy a neighbor if they're already all
                                  # distinct) -- see core814.apply_avoid_repeat. Directly
                                  # targets the same waste that w_triple penalizes. Raised
                                  # 10% -> 20% -> 34% (swapped with p_set_random) after real
                                  # runs showed dramatically faster score climbs from fresh
                                  # random seeds once this move and w_triple were introduced.
    p_remap_pair: float = 0.01   # swap 2 digits everywhere in the grid
    p_remap_full: float = 0.05  # relabel all 10 digits at once via a random permutation
                                 # (e.g. 0123456789 -> 2938475610) -- see core814.apply_remap_full
    p_edge_bias: float = 0.5     # probability a local move targets a border cell

    # --- temperature calibration ---------------------------------------------
    cal_samples: int = 2000
    p_hot: float = 0.5           # target acceptance rate at T0
    p_cold: float = 0.01         # target acceptance rate at T_end

    # --- schedule --------------------------------------------------------------
    cycle_iters: int = 2_000_000     # L_CYCLE: iterations per geometric-cooling cycle
    reheat: float = 0.6              # T <- T0 * reheat on stagnation-triggered reheat
    stagnation_iters: int = 500_000  # iters without a new best before reheating
    restart_from: str = "best"       # "best" or "current"; falls back after 3 stagnant cycles

    # --- parallel tempering ----------------------------------------------------
    swap_interval: int = 2000    # iterations between adjacent-replica swap attempts

    # --- checkpointing -----------------------------------------------------
    checkpoint_secs: float = 60.0
    print_secs: float = 5.0

    # --- misc --------------------------------------------------------------
    block_seconds: float = 0.25  # target wall-clock time per njit block (Ctrl-C latency)

    def move_probs(self) -> tuple:
        return (
            self.p_set_random,
            self.p_copy_neighbor,
            self.p_swap_adjacent,
            self.p_avoid_repeat,
            self.p_remap_pair,
            self.p_remap_full,
        )

    def cfg_hash(self) -> str:
        """Stable hash of the fields that affect run semantics (not run_name/resume/etc)."""
        semantic_fields = [
            "accept_mode", "search_mode", "lahc_len",
            "w_score", "w_look", "w_count", "w_heur", "w_triple", "want_count",
            "look_window", "count_lo", "count_hi",
            "p_set_random", "p_copy_neighbor", "p_swap_adjacent", "p_avoid_repeat", "p_remap_pair",
            "p_remap_full", "p_edge_bias", "replicas",
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

    p.add_argument("--checkpoint-secs", type=float, default=None)
    p.add_argument("--swap-interval", type=int, default=None)
    p.add_argument("--reheat", type=float, default=None)
    p.add_argument("--stagnation-iters", type=int, default=None)
    p.add_argument("--cycle-iters", type=int, default=None)
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

    if ns.fresh:
        cfg.resume = False
        cfg.fresh = True

    return cfg
