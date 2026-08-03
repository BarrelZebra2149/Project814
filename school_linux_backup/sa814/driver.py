"""
The outer (pure-Python) drive loop: owns the replica state arrays, calls
core814.run_block in short wall-clock-sized chunks, updates the temperature
schedule, triggers checkpoints, prints progress, and decides when to stop.

Split out from core814.py (rather than living inside it, as first sketched)
to avoid a circular import: seeding.py needs core814 to build dmasks/score
grids, and this module needs both core814 and seeding, so core814 itself
stays a leaf module with no dependency on the I/O/orchestration layer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numba
import numpy as np

import checkpoint
import core814 as core
import seeding
from config814 import SAConfig


def _log(msg: str) -> None:
    """sys.stdout.write(), not print(): skips print()'s per-call arg-join/sep/
    end handling for one flat string + newline. The real lever for perceived
    speed is usually how the output is consumed (redirect to a file instead
    of watching a live SSH/tmux terminal, which pays a render + round-trip
    cost per line) rather than the call itself, but this removes what
    overhead there is on the Python side too."""
    sys.stdout.write(msg + "\n")


def _geometric_ladder(t0: float, t_end: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([t0], dtype=np.float64)
    ratio = t_end / t0
    return np.array([t0 * (ratio ** (i / (n - 1))) for i in range(n)], dtype=np.float64)


def _energy_for(cfg: SAConfig, grid: np.ndarray, s: int, l: int, c: int) -> float:
    """energy_of() needs heur/triples, but three call sites (_fresh_state,
    _resumed_state, _restore_replica) were passing 0.0/0.0 regardless of
    cfg.w_heur/cfg.w_triple, so the energy they computed for scores_arr's
    initial/reseeded entries didn't match what run_block's kernel actually
    scores replicas by. Only pay for the (grid-wide) heur/triple scan when
    its weight is nonzero, matching how run_block already skips it."""
    heur = core.heur_chain_variance(grid) if cfg.w_heur != 0.0 else 0.0
    triples = float(core.count_triple_chains(grid)) if cfg.w_triple != 0.0 else 0.0
    return core.energy_of(s, l, c, heur, triples, cfg.w_score, cfg.w_look,
                           cfg.w_count, cfg.w_heur, cfg.w_triple)


def _fresh_verify_scratch() -> dict:
    """Standalone scratch buffers for _rescore(), independent of any
    replica's own dmask/stamp/gen -- so verification never shares (and can
    never be corrupted by) the same memo cache bug that caused it to be
    needed in the first place (see _restore_replica's stamps/gens fix)."""
    return dict(
        dmask=np.zeros((10, core.ROWS), dtype=np.int64),
        stamp=np.full(core.UPPER, -1, dtype=np.int64),
        buf=np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64),
        gen=0,
    )


def _rescore(cfg: SAConfig, grid: np.ndarray, scratch: dict):
    """Independently re-evaluates `grid` from scratch (fresh dmask, fresh
    gen so no stale stamp can be reused) instead of trusting scores_arr/
    energies, which are kernel-maintained bookkeeping that can desync from
    the actual grid contents (see _restore_replica's stale-memo bug)."""
    core.build_dmask(np.ascontiguousarray(grid), scratch["dmask"])
    scratch["gen"] += 1
    return core.evaluate(scratch["dmask"], scratch["stamp"], scratch["gen"], cfg.look_window,
                          cfg.count_lo, cfg.count_hi, cfg.want_count, scratch["buf"])


def _fresh_state(cfg: SAConfig, run_dir: Path, data_dir: Path, rng: np.random.Generator):
    R = cfg.replicas
    if cfg.seed_grids_file:
        n_parsed = len(seeding.parse_grids_from_file(Path(cfg.seed_grids_file)))
        grids = seeding.build_replicas_from_file(R, cfg.seed_grids_file, rng)
        scores = [seeding.score_grid(g) for g in grids]
        _log(f"[sa814] --seed-grids: {cfg.seed_grids_file} -> parsed {n_parsed} grids, "
              f"scores [{min(scores)}..{max(scores)}], seeding {R} replicas "
              f"({max(0, R - n_parsed)} perturbed)")
        if cfg.seed_from_corpus or cfg.seed_file:
            _log("[sa814] --seed-grids overrides --seed-file / corpus seeding; both ignored")
    else:
        grids = seeding.build_initial_replicas(R, data_dir, cfg.seed_file, run_dir, rng,
                                                use_corpus=cfg.seed_from_corpus)
    dmasks = np.zeros((R, 10, core.ROWS), dtype=np.int64)
    for i in range(R):
        core.build_dmask(grids[i], dmasks[i])
    stamps = np.full((R, core.UPPER), -1, dtype=np.int64)
    gens = np.ones(R, dtype=np.int64)
    digit_bufs = np.zeros((R, core.DIGIT_BUF_LEN), dtype=np.int64)
    rng_states = np.stack([core.make_rng_state(rng.integers(1, 2**62)) for _ in range(R)])

    scores_arr = np.zeros(R, dtype=np.int64)
    looks_arr = np.zeros(R, dtype=np.int64)
    counts_arr = np.zeros(R, dtype=np.int64)
    energies = np.zeros(R, dtype=np.float64)
    tmp_buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    for i in range(R):
        s, l, c = core.evaluate(dmasks[i], stamps[i], 1, cfg.look_window, cfg.count_lo,
                                 cfg.count_hi, cfg.want_count, tmp_buf)
        scores_arr[i], looks_arr[i], counts_arr[i] = s, l, c
        energies[i] = _energy_for(cfg, grids[i], s, l, c)

    t0, t_end, up_frac = core.calibrate_temperature(
        grids[0].copy(), dmasks[0].copy(), stamps[0].copy(), 1, rng_states[0].copy(),
        core.build_edge_positions(), np.array(cfg.move_probs()), cfg.p_edge_bias,
        cfg, n_samples=cfg.cal_samples, p_hot=cfg.p_hot, p_cold=cfg.p_cold,
    )
    _log(f"[sa814] temperature calibration: T0={t0:.5g} T_end={t_end:.5g} "
          f"(measured up-move rate {up_frac:.1%} on {cfg.cal_samples} samples)")

    if cfg.search_mode == "pt":
        temps = _geometric_ladder(t0, t_end, R)
    else:
        temps = np.full(R, t0, dtype=np.float64)

    hist = np.zeros((R, cfg.lahc_len), dtype=np.float64)
    for i in range(R):
        # + hist_reset_band, not exactly energies[i]: at exactly the
        # current energy, hmax == curE right after this fill, so DLAS
        # would accept only strict improvements (pure greedy, can't move)
        # until history re-diversifies. See config814.py's docstring.
        hist[i, :] = energies[i] + cfg.hist_reset_band
    lahc_pos = np.zeros(R, dtype=np.int64)
    cycle_pos = np.zeros(R, dtype=np.int64)
    accept_counter = np.zeros(R, dtype=np.int64)
    move_counter = np.zeros((R, core.N_MOVES), dtype=np.int64)
    move_accept_counter = np.zeros((R, core.N_MOVES), dtype=np.int64)
    swap_accept = np.zeros(max(R - 1, 0), dtype=np.int64)
    swap_attempt = np.zeros(max(R - 1, 0), dtype=np.int64)
    # Exact-state cycle prevention (core814.zobrist_hash / _anneal_one). Not
    # checkpoint-persisted -- it's a short-term recency window, not part of
    # the actual search state, so a fresh empty buffer on every process
    # start is harmless (equivalent to "nothing looks like a repeat yet").
    cycle_hashes = np.zeros((R, cfg.cycle_buffer), dtype=np.uint64)
    cycle_write_pos = np.zeros(R, dtype=np.int64)
    # Anchoring grace-period countdown (Fix 2): per-replica remaining
    # iterations of immunity from anchoring snapback after a reheat. Not
    # checkpoint-persisted -- same reasoning as cycle_hashes: a short-term
    # window, not real search state, so starting empty on every process
    # start is harmless (equivalent to "no grace owed yet").
    anchor_grace = np.zeros(R, dtype=np.int64)

    k_state = _fresh_k_state(R)

    best_idx = int(np.argmax(scores_arr))

    # Each replica starts out as its own personal best -- there's nothing
    # else to anchor to yet.
    pbest_grids = grids.copy()
    pbest_scores = scores_arr.copy()
    pbest_energies = energies.copy()

    state = dict(
        grids=grids, dmasks=dmasks, stamps=stamps, gens=gens,
        digit_bufs=digit_bufs, rng_states=rng_states,
        scores_arr=scores_arr, looks_arr=looks_arr, counts_arr=counts_arr, energies=energies,
        temps=temps, hist=hist, lahc_pos=lahc_pos, cycle_pos=cycle_pos,
        accept_counter=accept_counter, move_counter=move_counter,
        move_accept_counter=move_accept_counter,
        swap_accept=swap_accept, swap_attempt=swap_attempt,
        pbest_grids=pbest_grids, pbest_scores=pbest_scores, pbest_energies=pbest_energies,
        n_anchor_snaps=0, anchor_grace=anchor_grace,
        cycle_hashes=cycle_hashes, cycle_write_pos=cycle_write_pos,
        best_grid=grids[best_idx].copy(), best_score=int(scores_arr[best_idx]),
        best_energy=float(energies[best_idx]),
        total_iters=0, iters_since_best=0, elapsed_seconds=0.0, stagnant_cycles=0,
        t0=t0, t_end=t_end,
    )
    state.update(k_state)
    return state


def _fresh_k_state(R: int) -> dict:
    """--adaptive-k state: shared k-selection weights (all-ones = uniform,
    the default) plus per-replica attempt/accept counters and the decayed
    cross-block pooled totals they get folded into. See
    core814.weighted_index_choice and _update_k_weights."""
    return dict(
        k_weights_avoid=np.ones((2, 8), dtype=np.float64),
        k_weights_swap=np.ones((2, 8), dtype=np.float64),
        k_weights_remap=np.ones(9, dtype=np.float64),
        k_attempt_avoid=np.zeros((R, 2, 8), dtype=np.int64),
        k_accept_avoid=np.zeros((R, 2, 8), dtype=np.int64),
        k_attempt_swap=np.zeros((R, 2, 8), dtype=np.int64),
        k_accept_swap=np.zeros((R, 2, 8), dtype=np.int64),
        k_attempt_remap=np.zeros((R, 9), dtype=np.int64),
        k_accept_remap=np.zeros((R, 9), dtype=np.int64),
        k_pool_attempt_avoid=np.zeros((2, 8), dtype=np.float64),
        k_pool_accept_avoid=np.zeros((2, 8), dtype=np.float64),
        k_pool_attempt_swap=np.zeros((2, 8), dtype=np.float64),
        k_pool_accept_swap=np.zeros((2, 8), dtype=np.float64),
        k_pool_attempt_remap=np.zeros(9, dtype=np.float64),
        k_pool_accept_remap=np.zeros(9, dtype=np.float64),
        iters_since_k_update=0,
    )


def _update_k_weights(st: dict, cfg: SAConfig) -> None:
    """Pools this block's per-replica attempt/accept counts into the decayed
    running totals, recomputes k-selection weights from them (Laplace-
    smoothed acceptance rate: (accept + smoothing) / (attempt + 2*smoothing)),
    and resets the per-replica counters for the next block.
    weighted_index_choice only cares about relative magnitudes within the
    valid k-range at sampling time, so these don't need to sum to 1."""
    decay = cfg.adaptive_k_decay
    smoothing = cfg.adaptive_k_smoothing

    for name in ("avoid", "swap"):
        block_attempt = st[f"k_attempt_{name}"].sum(axis=0)
        block_accept = st[f"k_accept_{name}"].sum(axis=0)
        st[f"k_pool_attempt_{name}"] = st[f"k_pool_attempt_{name}"] * decay + block_attempt
        st[f"k_pool_accept_{name}"] = st[f"k_pool_accept_{name}"] * decay + block_accept
        st[f"k_weights_{name}"][:] = ((st[f"k_pool_accept_{name}"] + smoothing) /
                                      (st[f"k_pool_attempt_{name}"] + 2.0 * smoothing))
        st[f"k_attempt_{name}"][:] = 0
        st[f"k_accept_{name}"][:] = 0

    block_attempt = st["k_attempt_remap"].sum(axis=0)
    block_accept = st["k_accept_remap"].sum(axis=0)
    st["k_pool_attempt_remap"] = st["k_pool_attempt_remap"] * decay + block_attempt
    st["k_pool_accept_remap"] = st["k_pool_accept_remap"] * decay + block_accept
    st["k_weights_remap"][:] = ((st["k_pool_accept_remap"] + smoothing) /
                                (st["k_pool_attempt_remap"] + 2.0 * smoothing))
    st["k_attempt_remap"][:] = 0
    st["k_accept_remap"][:] = 0


def _resumed_state(ck: checkpoint.CheckpointState, cfg: SAConfig, rng: np.random.Generator):
    R = cfg.replicas
    if ck.grids.shape[0] != R:
        _log(f"[sa814] warning: checkpoint has {ck.grids.shape[0]} replicas, "
              f"config wants {R}; adjusting by truncating/padding.")
    n_common = min(R, ck.grids.shape[0])

    grids = np.empty((R, core.ROWS, core.COLS), dtype=np.uint8)
    grids[:n_common] = ck.grids[:n_common]
    for i in range(n_common, R):
        grids[i] = ck.best_grid.copy()

    dmasks = np.zeros((R, 10, core.ROWS), dtype=np.int64)
    for i in range(R):
        core.build_dmask(grids[i], dmasks[i])
    stamps = np.full((R, core.UPPER), -1, dtype=np.int64)
    gens = np.ones(R, dtype=np.int64)
    digit_bufs = np.zeros((R, core.DIGIT_BUF_LEN), dtype=np.int64)
    rng_states = np.empty((R, 2), dtype=np.uint64)
    rng_states[:n_common] = ck.rng_states[:n_common]
    for i in range(n_common, R):
        rng_states[i] = core.make_rng_state(rng.integers(1, 2**62))

    scores_arr = np.zeros(R, dtype=np.int64)
    looks_arr = np.zeros(R, dtype=np.int64)
    counts_arr = np.zeros(R, dtype=np.int64)
    energies = np.zeros(R, dtype=np.float64)
    tmp_buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    for i in range(R):
        s, l, c = core.evaluate(dmasks[i], stamps[i], 1, cfg.look_window, cfg.count_lo,
                                 cfg.count_hi, cfg.want_count, tmp_buf)
        scores_arr[i], looks_arr[i], counts_arr[i] = s, l, c
        energies[i] = _energy_for(cfg, grids[i], s, l, c)

    temps = np.empty(R, dtype=np.float64)
    temps[:n_common] = ck.temps[:n_common]
    t0 = float(np.max(ck.temps)) if ck.temps.size else 1.0
    t_end = float(np.min(ck.temps)) if ck.temps.size else 1e-6
    for i in range(n_common, R):
        temps[i] = t0

    lahc_len = ck.hist.shape[1] if ck.hist.ndim == 2 else cfg.lahc_len
    hist = np.empty((R, lahc_len), dtype=np.float64)
    hist[:n_common] = ck.hist[:n_common]
    # A history that was already frozen (all slots converged to curE, the
    # absorbing state Fix 1/min_worsening targets) at checkpoint time would
    # otherwise resume exactly as frozen as it was saved. Re-lay the same
    # floor a fresh start gets, so a resume can never be worse off than a
    # fresh run at recovering from this.
    hist[:n_common] = np.maximum(hist[:n_common], energies[:n_common, None] + cfg.hist_reset_band)
    for i in range(n_common, R):
        hist[i, :] = energies[i] + cfg.hist_reset_band
    lahc_pos = np.zeros(R, dtype=np.int64)
    lahc_pos[:n_common] = ck.lahc_pos[:n_common]
    cycle_pos = np.zeros(R, dtype=np.int64)
    cycle_pos[:n_common] = ck.cycle_pos[:n_common]
    accept_counter = np.zeros(R, dtype=np.int64)
    accept_counter[:n_common] = ck.accept_counter[:n_common]
    move_counter = np.zeros((R, core.N_MOVES), dtype=np.int64)
    # A checkpoint saved before a new move type was added may have fewer
    # columns (e.g. 5, before remap_full) -- carry over whatever overlaps
    # rather than erroring, since move_counter is just a stat, not real state.
    old_n_moves = min(ck.move_counter.shape[1], core.N_MOVES)
    move_counter[:n_common, :old_n_moves] = ck.move_counter[:n_common, :old_n_moves]
    move_accept_counter = np.zeros((R, core.N_MOVES), dtype=np.int64)
    old_n_moves_acc = min(ck.move_accept_counter.shape[1], core.N_MOVES)
    move_accept_counter[:n_common, :old_n_moves_acc] = ck.move_accept_counter[:n_common, :old_n_moves_acc]
    swap_accept = np.zeros(max(R - 1, 0), dtype=np.int64)
    swap_attempt = np.zeros(max(R - 1, 0), dtype=np.int64)
    # Not checkpoint-persisted (see _fresh_state's comment) -- always starts
    # fresh on resume too.
    cycle_hashes = np.zeros((R, cfg.cycle_buffer), dtype=np.uint64)
    cycle_write_pos = np.zeros(R, dtype=np.int64)
    # Anchoring grace countdown (Fix 2): not checkpoint-persisted, same as
    # cycle_hashes above -- always starts fresh on resume too.
    anchor_grace = np.zeros(R, dtype=np.int64)

    # Personal-best anchoring state: carry over whatever replicas overlap;
    # any newly-added replica (R grew) starts anchored to its own initial
    # grid, same cold start _fresh_state uses.
    pbest_grids = np.empty((R, core.ROWS, core.COLS), dtype=np.uint8)
    pbest_grids[:n_common] = ck.pbest_grids[:n_common]
    pbest_scores = np.zeros(R, dtype=np.int64)
    pbest_scores[:n_common] = ck.pbest_scores[:n_common]
    pbest_energies = np.zeros(R, dtype=np.float64)
    pbest_energies[:n_common] = ck.pbest_energies[:n_common]
    for i in range(n_common, R):
        pbest_grids[i] = grids[i].copy()
        pbest_scores[i] = scores_arr[i]
        pbest_energies[i] = energies[i]

    _log(f"[sa814] resumed run: total_iters={ck.total_iters} best_score={ck.best_score} "
          f"elapsed={ck.elapsed_seconds:.1f}s")

    # --adaptive-k: restore the learned weights + pooled totals (these aren't
    # per-replica, so an R change doesn't affect them); per-replica
    # attempt/accept counters always start fresh for the new block.
    k_state = _fresh_k_state(R)
    k_state["k_weights_avoid"] = ck.k_weights_avoid.copy()
    k_state["k_weights_swap"] = ck.k_weights_swap.copy()
    k_state["k_weights_remap"] = ck.k_weights_remap.copy()
    k_state["k_pool_attempt_avoid"] = ck.k_pool_attempt_avoid.copy()
    k_state["k_pool_accept_avoid"] = ck.k_pool_accept_avoid.copy()
    k_state["k_pool_attempt_swap"] = ck.k_pool_attempt_swap.copy()
    k_state["k_pool_accept_swap"] = ck.k_pool_accept_swap.copy()
    k_state["k_pool_attempt_remap"] = ck.k_pool_attempt_remap.copy()
    k_state["k_pool_accept_remap"] = ck.k_pool_accept_remap.copy()
    k_state["iters_since_k_update"] = int(ck.iters_since_k_update)

    state = dict(
        grids=grids, dmasks=dmasks, stamps=stamps, gens=gens,
        digit_bufs=digit_bufs, rng_states=rng_states,
        scores_arr=scores_arr, looks_arr=looks_arr, counts_arr=counts_arr, energies=energies,
        temps=temps, hist=hist, lahc_pos=lahc_pos, cycle_pos=cycle_pos,
        accept_counter=accept_counter, move_counter=move_counter,
        move_accept_counter=move_accept_counter,
        swap_accept=swap_accept, swap_attempt=swap_attempt,
        pbest_grids=pbest_grids, pbest_scores=pbest_scores, pbest_energies=pbest_energies,
        n_anchor_snaps=int(ck.n_anchor_snaps), anchor_grace=anchor_grace,
        cycle_hashes=cycle_hashes, cycle_write_pos=cycle_write_pos,
        best_grid=ck.best_grid.copy(), best_score=int(ck.best_score), best_energy=float(ck.best_energy),
        total_iters=int(ck.total_iters), iters_since_best=int(ck.iters_since_best),
        elapsed_seconds=float(ck.elapsed_seconds), stagnant_cycles=int(ck.stagnant_cycles),
        t0=t0, t_end=t_end,
    )
    state.update(k_state)
    return state


def _to_checkpoint_state(st: dict) -> checkpoint.CheckpointState:
    return checkpoint.CheckpointState(
        grids=st["grids"], energies=st["energies"], scores=st["scores_arr"],
        looks=st["looks_arr"], counts=st["counts_arr"], temps=st["temps"],
        rng_states=st["rng_states"], hist=st["hist"], lahc_pos=st["lahc_pos"],
        cycle_pos=st["cycle_pos"], accept_counter=st["accept_counter"],
        move_counter=st["move_counter"], swap_accept=st["swap_accept"],
        swap_attempt=st["swap_attempt"], best_grid=st["best_grid"],
        best_score=st["best_score"], best_energy=st["best_energy"],
        total_iters=st["total_iters"], iters_since_best=st["iters_since_best"],
        elapsed_seconds=st["elapsed_seconds"], stagnant_cycles=st["stagnant_cycles"],
        k_pool_attempt_avoid=st["k_pool_attempt_avoid"], k_pool_accept_avoid=st["k_pool_accept_avoid"],
        k_pool_attempt_swap=st["k_pool_attempt_swap"], k_pool_accept_swap=st["k_pool_accept_swap"],
        k_pool_attempt_remap=st["k_pool_attempt_remap"], k_pool_accept_remap=st["k_pool_accept_remap"],
        k_weights_avoid=st["k_weights_avoid"], k_weights_swap=st["k_weights_swap"],
        k_weights_remap=st["k_weights_remap"], iters_since_k_update=st["iters_since_k_update"],
        move_accept_counter=st["move_accept_counter"],
        pbest_grids=st["pbest_grids"], pbest_scores=st["pbest_scores"],
        pbest_energies=st["pbest_energies"], n_anchor_snaps=st["n_anchor_snaps"],
    )


def _restore_replica(st: dict, cfg: SAConfig, i: int, src_grid, kick_strength: int):
    """Resets replica i's grid to a copy of src_grid (e.g. st["best_grid"]
    or st["pbest_grids"][i]) if given, or leaves its current grid alone if
    src_grid is None (kick-in-place, used for restart_from="current" and
    "best"'s post-3-cycle fallback), then applies a kick_strength-cell
    random kick. Shared by the stagnation-triggered reheat block and the
    personal-best anchoring snapback (both need "reset toward some known
    grid + perturb + re-evaluate")."""
    if src_grid is not None:
        st["grids"][i] = src_grid.copy()
        core.build_dmask(st["grids"][i], st["dmasks"][i])
    if kick_strength > 0:
        core.apply_kick(st["rng_states"][i], st["grids"][i], st["dmasks"][i], kick_strength)
    # stamps[i]/gens[i] memoize which values were formable for the PREVIOUS
    # grid this replica held. That used to only get invalidated inside
    # `if from_best:`, so a kick-only reseed (from_best=False) fed evaluate()
    # the old grid's memo against the new (kicked) grid -- every value the
    # old grid could form got treated as still-formable without being
    # retested, inflating the returned score by up to look_window. Always
    # invalidate after any grid mutation, before evaluate() below.
    st["stamps"][i, :] = -1
    st["gens"][i] = 1
    tmp_buf = st["digit_bufs"][i]
    s, l, c = core.evaluate(st["dmasks"][i], st["stamps"][i], st["gens"][i], cfg.look_window,
                             cfg.count_lo, cfg.count_hi, cfg.want_count, tmp_buf)
    st["scores_arr"][i], st["looks_arr"][i], st["counts_arr"][i] = s, l, c
    st["energies"][i] = _energy_for(cfg, st["grids"][i], s, l, c)
    st["hist"][i, :] = st["energies"][i] + cfg.hist_reset_band
    st["lahc_pos"][i] = 0
    # Cycle buffer holds "recently visited states of this grid's current
    # lineage" -- after a restore (reheat or anchor snapback), that lineage
    # has effectively restarted, so stale entries from wherever it just was
    # would only risk rare false-positive rejections later. Cheap to clear.
    st["cycle_hashes"][i, :] = 0
    st["cycle_write_pos"][i] = 0


def _update_elite_pool(st: dict, cfg: SAConfig) -> None:
    """Maintains up to cfg.elite_size distinct top grids (by score, deduped
    on raw bytes) drawn from the replicas' own personal bests, feeding
    _elite_resample's "go with the winners" step. Off (elite_resample_iters
    == 0) by default -- a second, compounding diversity mechanism on top of
    per-replica anchoring, meant to be enabled only after measuring
    anchoring alone."""
    pool = st["elite_pool"]
    seen = {g.tobytes() for _, g in pool}
    for i in range(cfg.replicas):
        key = st["pbest_grids"][i].tobytes()
        if key in seen:
            continue
        pool.append((int(st["pbest_scores"][i]), st["pbest_grids"][i].copy()))
        seen.add(key)
    pool.sort(key=lambda t: -t[0])
    del pool[cfg.elite_size:]


def _elite_resample(st: dict, cfg: SAConfig, rng: np.random.Generator, verify_scratch: dict) -> None:
    """Reassigns the worst-performing quarter of replicas (by pbest_score)
    a uniformly random grid from the elite pool, then kicks and
    re-anneals from there."""
    pool = st["elite_pool"]
    if not pool:
        return
    order = np.argsort(st["pbest_scores"])
    n_replace = max(1, cfg.replicas // 4)
    for idx in order[:n_replace]:
        i = int(idx)
        _, elite_grid = pool[int(rng.integers(0, len(pool)))]
        _restore_replica(st, cfg, i, elite_grid, cfg.anchor_kick)
        # elite_grid (pre-kick) is already known-good -- re-verify it
        # directly rather than trusting _restore_replica's post-kick
        # evaluate() above for the pbest bookkeeping, since the kick may
        # have made the live grid worse than the elite grid it came from.
        e_s, e_l, e_c = _rescore(cfg, elite_grid, verify_scratch)
        e_e = _energy_for(cfg, elite_grid, e_s, e_l, e_c)
        pb_s = int(st["pbest_scores"][i])
        if e_s > pb_s or (e_s == pb_s and e_e < float(st["pbest_energies"][i])):
            st["pbest_grids"][i] = elite_grid.copy()
            st["pbest_scores"][i] = e_s
            st["pbest_energies"][i] = e_e


def _safe_checkpoint_io(action: str, fn, *args, **kwargs) -> None:
    """Runs a checkpoint.py I/O call, catching OSError so a missed write
    never crashes an otherwise-healthy, possibly hours-long search. On
    Windows, os.replace() can fail with a transient PermissionError even
    after checkpoint.py's own short retry loop, if something (a real-time
    antivirus scan, a PyCharm/editor indexer watching the project folder, a
    cloud-sync client) holds an unusually long lock on the destination file.
    Losing one write just means the next periodic/new-record attempt tries
    again -- far better than losing the whole run."""
    try:
        fn(*args, **kwargs)
    except OSError as exc:
        _log(f"[sa814] warning: {action} failed ({exc!r}); will retry next time")


def drive(cfg: SAConfig, runtime, base_dir: Path) -> None:
    run_dir = checkpoint.run_root(base_dir, cfg.run_name)
    cfg_hash = cfg.cfg_hash()
    data_dir = runtime.data_dir()
    rng = np.random.default_rng(cfg.rng_seed)

    threads = cfg.threads or runtime.default_threads()
    numba.set_num_threads(threads)
    _log(f"[sa814] run='{cfg.run_name}' mode={cfg.search_mode} accept={cfg.accept_mode} "
          f"replicas={cfg.replicas} threads={threads} cfg_hash={cfg_hash}")
    if cfg.rng_seed is None:
        _log("[sa814] warning: no --rng-seed given; this run is not reproducible "
              "(two runs with identical flags will still diverge). Pass --rng-seed "
              "<int> for comparable A/B runs.")
    else:
        _log(f"[sa814] rng_seed={cfg.rng_seed}")
    if not cfg.seed_from_corpus:
        _log("[sa814] --no-seed: ignoring data/*.txt and any prior run outputs, "
              "starting every replica from a fresh random grid")
    if cfg.adaptive_k:
        _log(f"[sa814] --adaptive-k: learning avoid_repeat/swap_adjacent/remap's k-distribution "
              f"from observed accept rates (update every {cfg.adaptive_k_update_iters} iters, "
              f"decay={cfg.adaptive_k_decay}, smoothing={cfg.adaptive_k_smoothing})")
    if cfg.anchor_enabled and cfg.anchor_grace_iters >= cfg.stagnation_iters:
        # anchor_grace_iters and iters_since_best are both aggregate-iter
        # counters (see the anchoring pass below), so reheat re-arms grace
        # to anchor_grace_iters every stagnation_iters aggregate iters. If
        # anchor_grace_iters >= stagnation_iters, grace never has a chance
        # to expire before the next reheat re-arms it -- anchoring's
        # snapback branch becomes permanently unreachable, silently
        # disabling the whole mechanism (confirmed on a real production
        # run: n_snaps froze immediately after the first reheat while
        # anchor_gap grew past 4000 over 6.7 hours). Clamp rather than
        # just warn, since a silently-broken default is worse than a
        # loud auto-correction.
        clamped = cfg.stagnation_iters // 2
        _log(f"[sa814] WARNING: anchor_grace_iters({cfg.anchor_grace_iters}) >= "
              f"stagnation_iters({cfg.stagnation_iters}) -- reheat would re-arm anchoring's "
              f"grace period faster than it can expire, permanently disabling snapback. "
              f"Clamping anchor_grace_iters to {clamped}.")
        cfg.anchor_grace_iters = clamped

    verify_scratch = _fresh_verify_scratch()

    st = None
    if cfg.resume and not cfg.fresh:
        ck = checkpoint.load(run_dir)
        meta = checkpoint.load_meta(run_dir)
        if ck is not None:
            if meta is not None and meta.get("cfg_hash") != cfg_hash:
                _log(f"[sa814] checkpoint cfg_hash mismatch "
                      f"({meta.get('cfg_hash')} != {cfg_hash}); reseeding fresh from its best grid only.")
                st = _fresh_state(cfg, run_dir, data_dir, rng)
                # Don't trust ck.best_score/best_energy blindly -- re-verify
                # against the actual stored grid before letting it override
                # the freshly-seeded best (see _rescore's docstring).
                v_s, v_l, v_c = _rescore(cfg, ck.best_grid, verify_scratch)
                v_e = _energy_for(cfg, ck.best_grid, v_s, v_l, v_c)
                if v_s != int(ck.best_score):
                    _log(f"[sa814] warning: checkpoint best_grid claimed score={ck.best_score} "
                          f"but verified score={v_s}; using verified value.")
                if v_s > st["best_score"]:
                    st["best_grid"] = ck.best_grid.copy()
                    st["best_score"] = v_s
                    st["best_energy"] = v_e
                    st["grids"][0] = ck.best_grid.copy()
                    core.build_dmask(st["grids"][0], st["dmasks"][0])
            else:
                st = _resumed_state(ck, cfg, rng)

    if st is None:
        st = _fresh_state(cfg, run_dir, data_dir, rng)

    # One-time startup verification: st["best_grid"] came from either a
    # fresh seed's scores_arr/energies bookkeeping or a resumed checkpoint's
    # stored best_score/best_energy, neither of which is re-checked against
    # the grid itself elsewhere. Do it once here so write_best below (and
    # every later comparison against st["best_score"]) starts from a
    # verified baseline.
    v_s, v_l, v_c = _rescore(cfg, st["best_grid"], verify_scratch)
    v_e = _energy_for(cfg, st["best_grid"], v_s, v_l, v_c)
    if v_s != st["best_score"]:
        _log(f"[sa814] warning: startup best_grid claimed score={st['best_score']} "
              f"but verified score={v_s}; correcting.")
    st["best_score"] = v_s
    st["best_energy"] = v_e
    st["desync_count"] = 0
    # Elite pool for --elite-resample-iters (off by default): not
    # checkpointed, always starts empty and gets rebuilt from pbest_grids
    # within one _update_elite_pool call -- cheap enough not to bother
    # persisting.
    st["elite_pool"] = []
    st["iters_since_elite_resample"] = 0

    edge_pos = core.build_edge_positions()
    move_probs = np.array(cfg.move_probs(), dtype=np.float64)
    move_probs = move_probs / move_probs.sum()
    swap_rng = core.make_rng_state(rng.integers(1, 2**62))
    accept_mode = core.ACCEPT_MODE_CODE[cfg.accept_mode]
    do_swaps = (cfg.search_mode == "pt") and cfg.replicas > 1

    stop_flag = {"stop": False}
    runtime.install_handlers(lambda: stop_flag.__setitem__("stop", True))

    wall_start = time.perf_counter() - st["elapsed_seconds"]
    last_checkpoint = time.perf_counter()
    last_print = time.perf_counter()
    iters_per_segment = 2000
    n_segments = 5
    measured_iters_per_sec = None
    # Fix 3: accept_counter/move_counter are cumulative for the whole run
    # (never reset, carried across checkpoints), so the printed accept rate
    # kept showing 3.9-5.3% on servers that had actually frozen solid --
    # already-accumulated hundreds of millions of iters' worth of history
    # swamped a recent true rate of 0%. Snapshot the previous print's totals
    # and the previous scores_arr so each print can report the DELTA since
    # last time (accept_recent, n_moved) alongside the old cumulative value
    # (renamed accept_life for clarity).
    prev_total_accept = 0
    prev_total_moves = 0
    prev_scores_snapshot = st["scores_arr"].copy()

    def save_now():
        _safe_checkpoint_io("checkpoint save", checkpoint.save, run_dir,
                             _to_checkpoint_state(st), cfg.to_dict(), cfg_hash)

    # Ensure best.txt always reflects the best-known grid, even if this
    # session never beats it: a fresh run seeded from an already-strong
    # corpus grid (or a resumed run that stalls) would otherwise never write
    # best.txt at all, since that only happened on a *new* record before.
    _safe_checkpoint_io("write_best", checkpoint.write_best, run_dir, st["best_grid"], st["best_score"])

    _log(f"[sa814] starting from best_score={st['best_score']} total_iters={st['total_iters']}")

    while True:
        t_block = time.perf_counter()
        core.run_block(
            st["grids"], st["dmasks"], st["stamps"], st["gens"], st["energies"],
            st["scores_arr"], st["looks_arr"], st["counts_arr"], st["temps"], st["rng_states"],
            st["digit_bufs"],
            st["hist"], st["lahc_pos"], swap_rng,
            edge_pos, move_probs, cfg.p_edge_bias,
            cfg.w_score, cfg.w_look, cfg.w_count, cfg.w_heur, cfg.w_triple, cfg.want_count,
            cfg.look_window, cfg.count_lo, cfg.count_hi,
            accept_mode, iters_per_segment, n_segments, do_swaps, cfg.max_worsening, cfg.min_worsening,
            st["accept_counter"], st["move_counter"], st["move_accept_counter"],
            st["swap_accept"], st["swap_attempt"],
            st["k_weights_avoid"], st["k_weights_swap"], st["k_weights_remap"],
            st["k_attempt_avoid"], st["k_accept_avoid"], st["k_attempt_swap"], st["k_accept_swap"],
            st["k_attempt_remap"], st["k_accept_remap"],
            core.ZOBRIST, st["cycle_hashes"], st["cycle_write_pos"],
        )
        block_elapsed = time.perf_counter() - t_block
        iters_done = cfg.replicas * iters_per_segment * n_segments
        st["total_iters"] += iters_done
        st["elapsed_seconds"] = time.perf_counter() - wall_start

        if cfg.adaptive_k:
            st["iters_since_k_update"] += iters_done
            if st["iters_since_k_update"] >= cfg.adaptive_k_update_iters:
                _update_k_weights(st, cfg)
                st["iters_since_k_update"] = 0
                best_k_avoid = [int(np.argmax(st["k_weights_avoid"][p])) + 1 for p in (0, 1)]
                best_k_swap = [int(np.argmax(st["k_weights_swap"][p])) + 1 for p in (0, 1)]
                best_k_remap = int(np.argmax(st["k_weights_remap"])) + 2
                _log(f"[sa814] adaptive-k updated (iter {st['total_iters']}): "
                      f"avoid_repeat k*=[interior={best_k_avoid[0]}, border={best_k_avoid[1]}]  "
                      f"swap_adjacent k*=[interior={best_k_swap[0]}, border={best_k_swap[1]}]  "
                      f"remap k*={best_k_remap}")

        measured_iters_per_sec = iters_done / max(block_elapsed, 1e-6)
        if block_elapsed > 0:
            target_total_iters = max(int(measured_iters_per_sec * cfg.block_seconds), cfg.replicas)
            n_segments = max(1, target_total_iters // (cfg.replicas * iters_per_segment))

        cur_best_idx = int(np.argmax(st["scores_arr"]))
        # cur_best_score/cur_best_energy are used below both for the periodic
        # log line (informational, every block) and for the promotion check
        # (correctness-critical, re-verified below when a promotion looks
        # possible) -- always set from the claimed values first so the log
        # line has something current even on blocks with no promotion.
        cur_best_score = int(st["scores_arr"][cur_best_idx])
        cur_best_energy = float(st["energies"][cur_best_idx])
        claim_score, claim_energy = cur_best_score, cur_best_energy
        # energy_of() is lower-is-better (core814.energy_of returns a
        # negated goodness), so the tie-break must prefer the LOWER energy.
        # This used to be `>`, which on an exact score tie replaced
        # best_grid with the worse of the two grids.
        promotable = (claim_score > st["best_score"] or
                      (claim_score == st["best_score"] and claim_energy < st["best_energy"]))
        if promotable:
            # scores_arr/energies are kernel-maintained bookkeeping that can
            # desync from the grid itself (see _restore_replica's stale-memo
            # fix) -- re-verify independently before promoting/persisting,
            # and self-correct the bookkeeping either way so a stale desync
            # doesn't keep re-triggering every block.
            v_s, v_l, v_c = _rescore(cfg, st["grids"][cur_best_idx], verify_scratch)
            v_e = _energy_for(cfg, st["grids"][cur_best_idx], v_s, v_l, v_c)
            if v_s != claim_score or abs(v_e - claim_energy) > 1e-6:
                st["desync_count"] += 1
                _log(f"[sa814] warning: replica {cur_best_idx} score desync "
                      f"(kernel claimed score={claim_score} energy={claim_energy:.3f}; "
                      f"verified score={v_s} energy={v_e:.3f}); self-correcting "
                      f"(desync_count={st['desync_count']}).")
                st["scores_arr"][cur_best_idx] = v_s
                st["looks_arr"][cur_best_idx] = v_l
                st["counts_arr"][cur_best_idx] = v_c
                st["energies"][cur_best_idx] = v_e
            cur_best_score, cur_best_energy = v_s, v_e
            promotable = (cur_best_score > st["best_score"] or
                          (cur_best_score == st["best_score"] and cur_best_energy < st["best_energy"]))
        if promotable:
            st["best_score"] = cur_best_score
            st["best_energy"] = cur_best_energy
            st["best_grid"] = st["grids"][cur_best_idx].copy()
            st["iters_since_best"] = 0
            st["stagnant_cycles"] = 0
            _log(f"[sa814] NEW RECORD: score={cur_best_score} "
                  f"(iter {st['total_iters']}, t={st['elapsed_seconds']:.1f}s)")
            _safe_checkpoint_io("write_best", checkpoint.write_best, run_dir,
                                 st["best_grid"], st["best_score"])
            _safe_checkpoint_io("append_record", checkpoint.append_record, run_dir,
                                 st["best_grid"], st["best_score"],
                                 st["total_iters"], st["elapsed_seconds"])
            save_now()
            last_checkpoint = time.perf_counter()
        else:
            st["iters_since_best"] += iters_done

        # Personal-best anchoring: real runs showed replicas random-walking
        # for the whole run at scores far below best_score (e.g. best=6549,
        # live replicas 334-1101) once a catastrophic move collapsed one --
        # nothing pulled them back. Track each replica's own best (not the
        # global best, which would collapse every replica onto one basin --
        # exactly what stagnation-triggered reheat already does) and snap
        # it back once it's drifted anchor_margin points below its own
        # pbest, every block rather than only at stagnation_iters
        # intervals. Trusts scores_arr/energies directly (not a fresh
        # _rescore) since _restore_replica's memo-invalidation fix means
        # they can no longer desync from the grid during normal annealing.
        if cfg.anchor_enabled:
            # Fix 2: a reheat's kick (see the stagnation/reheat block below)
            # gets exactly one block to prove itself before this pass ran --
            # since it almost always collapses a high-scoring grid outright
            # (median positive dE near a 7666 grid is ~6129, see anchor_kick's
            # docstring), the very next anchoring pass used to see "score <
            # pbest - anchor_margin" and snap straight back, undoing the kick
            # before it ever got a chance to explore (pbest is monotone, so
            # it never falls to meet the kicked replica partway). anchor_grace
            # is a countdown of immunity from snapback set right after a
            # reheat restore; it's independent of -- and always overridden
            # by -- an actual improvement, which must be recorded the
            # instant it happens regardless of grace.
            #
            # Unit bug fixed here: `iters_since_best`/`stagnation_iters`
            # (which drives how often reheat re-arms grace, below) are
            # counted in AGGREGATE iters (`iters_done` is already
            # `replicas * iters_per_segment * n_segments`), but this used
            # to decrement grace by `iters_done // replicas` (per-replica
            # units). With R replicas, reheat re-arms grace to
            # anchor_grace_iters every `stagnation_iters` aggregate iters =
            # `stagnation_iters / R` PER-REPLICA iters, while burning grace
            # down from anchor_grace_iters took `anchor_grace_iters`
            # per-replica iters -- R times slower than it needed to be. At
            # the default anchor_grace_iters=200_000, stagnation_iters=
            # 500_000, this made grace permanently non-expiring for any
            # R >= 2 (confirmed on a real 16-replica production run:
            # n_snaps froze immediately after the first reheat while
            # anchor_gap grew past 4000 -- anchoring was completely inert
            # for the entire 6.7-hour run). Decrementing by `iters_done`
            # (the same aggregate unit `anchor_grace_iters` is re-armed and
            # compared in) fixes this regardless of replica count.
            for i in range(cfg.replicas):
                s_i = int(st["scores_arr"][i])
                e_i = float(st["energies"][i])
                pb_s = int(st["pbest_scores"][i])
                if s_i > pb_s or (s_i == pb_s and e_i < st["pbest_energies"][i]):
                    st["pbest_grids"][i] = st["grids"][i].copy()
                    st["pbest_scores"][i] = s_i
                    st["pbest_energies"][i] = e_i
                    st["anchor_grace"][i] = 0
                elif st["anchor_grace"][i] > 0:
                    st["anchor_grace"][i] -= iters_done
                elif s_i < pb_s - cfg.anchor_margin:
                    _restore_replica(st, cfg, i, st["pbest_grids"][i], cfg.anchor_kick)
                    st["n_anchor_snaps"] += 1

        if cfg.elite_resample_iters > 0:
            _update_elite_pool(st, cfg)
            st["iters_since_elite_resample"] += iters_done
            if st["iters_since_elite_resample"] >= cfg.elite_resample_iters:
                _elite_resample(st, cfg, rng, verify_scratch)
                st["iters_since_elite_resample"] = 0

        if cfg.search_mode == "anneal":
            for i in range(cfg.replicas):
                st["cycle_pos"][i] += iters_done
                frac = min(1.0, st["cycle_pos"][i] / cfg.cycle_iters)
                st["temps"][i] = st["t0"] * ((st["t_end"] / st["t0"]) ** frac)

        if st["iters_since_best"] >= cfg.stagnation_iters:
            st["stagnant_cycles"] += 1
            # config814.py's restart_from docstring: "best" restarts every
            # replica from the global best_grid each reheat, but falls back
            # to kicking the replica's own current state after 3
            # consecutive stagnant cycles (to add diversity once repeatedly
            # returning to best isn't escaping the plateau); "pbest"
            # (default) restarts each replica from its OWN best grid every
            # time -- already diverse across replicas since each one has
            # its own history, so it doesn't need the 3-cycle fallback;
            # "current" never restarts from any stored grid, always
            # kicking in place.
            use_best = cfg.restart_from == "best" and st["stagnant_cycles"] < 3
            _log(f"[sa814] stagnation ({st['iters_since_best']} iters without improvement) -> "
                  f"reheating (stagnant_cycles={st['stagnant_cycles']}, "
                  f"restart_from={cfg.restart_from})")
            for i in range(cfg.replicas):
                kick = 2 + (i % 6)
                if cfg.restart_from == "pbest":
                    src_grid = st["pbest_grids"][i]
                elif use_best:
                    src_grid = st["best_grid"]
                else:
                    src_grid = None
                _restore_replica(st, cfg, i, src_grid, kick_strength=kick)
                st["anchor_grace"][i] = cfg.anchor_grace_iters
                if cfg.search_mode == "anneal":
                    st["cycle_pos"][i] = 0
                    st["temps"][i] = st["t0"] * cfg.reheat
            if cfg.search_mode == "pt":
                # re-derive the ladder around the reheated top temperature so the
                # spread of exploration/exploitation is preserved after reseeding
                st["temps"] = _geometric_ladder(st["t0"] * cfg.reheat, st["t_end"], cfg.replicas)
            if st["stagnant_cycles"] >= 3:
                st["stagnant_cycles"] = 0
            st["iters_since_best"] = 0

        now = time.perf_counter()
        if now - last_checkpoint >= cfg.checkpoint_secs:
            save_now()
            last_checkpoint = now

        if now - last_print >= cfg.print_secs:
            mean_e = float(np.mean(st["energies"]))
            total_accept = int(np.sum(st["accept_counter"]))
            total_moves = int(np.sum(st["move_counter"]))
            accept_life = total_accept / max(total_moves, 1)
            recent_accept = total_accept - prev_total_accept
            recent_moves = total_moves - prev_total_moves
            accept_recent = recent_accept / max(recent_moves, 1)
            # Number of replicas whose score changed since the last print --
            # a direct, unambiguous "is the search actually moving" signal.
            # A frozen server (all replica scores bit-for-bit unchanged for
            # minutes) shows n_moved=0 every single print, which accept_life
            # alone could never surface once enough history had accumulated.
            n_moved = int(np.sum(st["scores_arr"] != prev_scores_snapshot))
            # Diagnostic for how far the live replica population has
            # drifted from best_grid -- this is what exposed the solver
            # random-walking near score ~1000 while best_score sat at
            # 5000+ (see the stagnation investigation). Not gated behind
            # any flag: it's just a median() and a subtraction.
            rep_score_min = int(np.min(st["scores_arr"]))
            rep_score_med = float(np.median(st["scores_arr"]))
            rep_score_max = int(np.max(st["scores_arr"]))
            anchor_gap = st["best_score"] - rep_score_med
            _log(f"[sa814] iter={st['total_iters']:>12d}  t={st['elapsed_seconds']:>7.1f}s  "
                  f"best={st['best_score']:>5d}  cur_best={cur_best_score:>5d}  "
                  f"rep_score=[{rep_score_min},{rep_score_med:.0f},{rep_score_max}]  "
                  f"n_moved={n_moved:>3d}  "
                  f"anchor_gap={anchor_gap:.0f}  n_snaps={st['n_anchor_snaps']:>4d}  "
                  f"mean_E={mean_e:>10.2f}  accept_recent={accept_recent:>5.1%}  accept_life={accept_life:>5.1%}  "
                  f"T=[{np.min(st['temps']):.4g},{np.max(st['temps']):.4g}]  "
                  f"{measured_iters_per_sec:>10.0f} it/s")
            _safe_checkpoint_io("append_progress", checkpoint.append_progress, run_dir, {
                "iter": st["total_iters"], "elapsed": round(st["elapsed_seconds"], 1),
                "best_score": st["best_score"], "mean_energy": round(mean_e, 3),
                "accept_life": round(accept_life, 4), "accept_recent": round(accept_recent, 4),
                "n_moved": n_moved,
                "T_min": float(np.min(st["temps"])), "T_max": float(np.max(st["temps"])),
                "rep_score_min": rep_score_min, "rep_score_med": rep_score_med,
                "rep_score_max": rep_score_max, "anchor_gap": round(anchor_gap, 1),
                "n_snaps": st["n_anchor_snaps"], "desyncs": st["desync_count"],
            })
            prev_total_accept = total_accept
            prev_total_moves = total_moves
            prev_scores_snapshot = st["scores_arr"].copy()
            last_print = now

        if cfg.max_seconds is not None and st["elapsed_seconds"] >= cfg.max_seconds:
            _log(f"[sa814] stopping: reached max_seconds={cfg.max_seconds}")
            break
        if cfg.max_iters is not None and st["total_iters"] >= cfg.max_iters:
            _log(f"[sa814] stopping: reached max_iters={cfg.max_iters}")
            break
        if cfg.target_score is not None and st["best_score"] >= cfg.target_score:
            _log(f"[sa814] stopping: reached target_score={cfg.target_score}")
            break
        if stop_flag["stop"]:
            _log("[sa814] stopping: signal received")
            break

    _safe_checkpoint_io("write_best", checkpoint.write_best, run_dir, st["best_grid"], st["best_score"])
    save_now()
    _log(f"[sa814] final: best_score={st['best_score']} total_iters={st['total_iters']} "
          f"elapsed={st['elapsed_seconds']:.1f}s -> {run_dir / 'best.txt'}")
