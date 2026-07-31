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

import time
from pathlib import Path

import numba
import numpy as np

import checkpoint
import core814 as core
import seeding
from config814 import SAConfig


def _geometric_ladder(t0: float, t_end: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([t0], dtype=np.float64)
    ratio = t_end / t0
    return np.array([t0 * (ratio ** (i / (n - 1))) for i in range(n)], dtype=np.float64)


def _fresh_state(cfg: SAConfig, run_dir: Path, data_dir: Path, rng: np.random.Generator):
    R = cfg.replicas
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
        energies[i] = core.energy_of(s, l, c, 0.0, 0.0, cfg.w_score, cfg.w_look, cfg.w_count,
                                      cfg.w_heur, cfg.w_triple)

    t0, t_end, up_frac = core.calibrate_temperature(
        grids[0].copy(), dmasks[0].copy(), stamps[0].copy(), 1, rng_states[0].copy(),
        core.build_edge_positions(), np.array(cfg.move_probs()), cfg.p_edge_bias,
        cfg, n_samples=cfg.cal_samples, p_hot=cfg.p_hot, p_cold=cfg.p_cold,
    )
    print(f"[sa814] temperature calibration: T0={t0:.5g} T_end={t_end:.5g} "
          f"(measured up-move rate {up_frac:.1%} on {cfg.cal_samples} samples)")

    if cfg.search_mode == "pt":
        temps = _geometric_ladder(t0, t_end, R)
    else:
        temps = np.full(R, t0, dtype=np.float64)

    hist = np.zeros((R, cfg.lahc_len), dtype=np.float64)
    for i in range(R):
        hist[i, :] = energies[i]
    lahc_pos = np.zeros(R, dtype=np.int64)
    cycle_pos = np.zeros(R, dtype=np.int64)
    accept_counter = np.zeros(R, dtype=np.int64)
    move_counter = np.zeros((R, core.N_MOVES), dtype=np.int64)
    swap_accept = np.zeros(max(R - 1, 0), dtype=np.int64)
    swap_attempt = np.zeros(max(R - 1, 0), dtype=np.int64)

    best_idx = int(np.argmax(scores_arr))

    state = dict(
        grids=grids, dmasks=dmasks, stamps=stamps, gens=gens,
        digit_bufs=digit_bufs, rng_states=rng_states,
        scores_arr=scores_arr, looks_arr=looks_arr, counts_arr=counts_arr, energies=energies,
        temps=temps, hist=hist, lahc_pos=lahc_pos, cycle_pos=cycle_pos,
        accept_counter=accept_counter, move_counter=move_counter,
        swap_accept=swap_accept, swap_attempt=swap_attempt,
        best_grid=grids[best_idx].copy(), best_score=int(scores_arr[best_idx]),
        best_energy=float(energies[best_idx]),
        total_iters=0, iters_since_best=0, elapsed_seconds=0.0, stagnant_cycles=0,
        t0=t0, t_end=t_end,
    )
    return state


def _resumed_state(ck: checkpoint.CheckpointState, cfg: SAConfig, rng: np.random.Generator):
    R = cfg.replicas
    if ck.grids.shape[0] != R:
        print(f"[sa814] warning: checkpoint has {ck.grids.shape[0]} replicas, "
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
        energies[i] = core.energy_of(s, l, c, 0.0, 0.0, cfg.w_score, cfg.w_look, cfg.w_count,
                                      cfg.w_heur, cfg.w_triple)

    temps = np.empty(R, dtype=np.float64)
    temps[:n_common] = ck.temps[:n_common]
    t0 = float(np.max(ck.temps)) if ck.temps.size else 1.0
    t_end = float(np.min(ck.temps)) if ck.temps.size else 1e-6
    for i in range(n_common, R):
        temps[i] = t0

    lahc_len = ck.hist.shape[1] if ck.hist.ndim == 2 else cfg.lahc_len
    hist = np.empty((R, lahc_len), dtype=np.float64)
    hist[:n_common] = ck.hist[:n_common]
    for i in range(n_common, R):
        hist[i, :] = energies[i]
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
    swap_accept = np.zeros(max(R - 1, 0), dtype=np.int64)
    swap_attempt = np.zeros(max(R - 1, 0), dtype=np.int64)

    print(f"[sa814] resumed run: total_iters={ck.total_iters} best_score={ck.best_score} "
          f"elapsed={ck.elapsed_seconds:.1f}s")

    return dict(
        grids=grids, dmasks=dmasks, stamps=stamps, gens=gens,
        digit_bufs=digit_bufs, rng_states=rng_states,
        scores_arr=scores_arr, looks_arr=looks_arr, counts_arr=counts_arr, energies=energies,
        temps=temps, hist=hist, lahc_pos=lahc_pos, cycle_pos=cycle_pos,
        accept_counter=accept_counter, move_counter=move_counter,
        swap_accept=swap_accept, swap_attempt=swap_attempt,
        best_grid=ck.best_grid.copy(), best_score=int(ck.best_score), best_energy=float(ck.best_energy),
        total_iters=int(ck.total_iters), iters_since_best=int(ck.iters_since_best),
        elapsed_seconds=float(ck.elapsed_seconds), stagnant_cycles=int(ck.stagnant_cycles),
        t0=t0, t_end=t_end,
    )


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
    )


def _reseed_replica(st: dict, cfg: SAConfig, i: int, from_best: bool, kick_strength: int):
    if from_best:
        st["grids"][i] = st["best_grid"].copy()
        core.build_dmask(st["grids"][i], st["dmasks"][i])
        st["stamps"][i, :] = -1
        st["gens"][i] = 1
    if kick_strength > 0:
        core.apply_kick(st["rng_states"][i], st["grids"][i], st["dmasks"][i], kick_strength)
    tmp_buf = st["digit_bufs"][i]
    s, l, c = core.evaluate(st["dmasks"][i], st["stamps"][i], st["gens"][i], cfg.look_window,
                             cfg.count_lo, cfg.count_hi, cfg.want_count, tmp_buf)
    st["scores_arr"][i], st["looks_arr"][i], st["counts_arr"][i] = s, l, c
    st["energies"][i] = core.energy_of(s, l, c, 0.0, 0.0, cfg.w_score, cfg.w_look, cfg.w_count,
                                        cfg.w_heur, cfg.w_triple)
    st["hist"][i, :] = st["energies"][i]
    st["lahc_pos"][i] = 0


def drive(cfg: SAConfig, runtime, base_dir: Path) -> None:
    run_dir = checkpoint.run_root(base_dir, cfg.run_name)
    cfg_hash = cfg.cfg_hash()
    data_dir = runtime.data_dir()
    rng = np.random.default_rng()

    threads = cfg.threads or runtime.default_threads()
    numba.set_num_threads(threads)
    print(f"[sa814] run='{cfg.run_name}' mode={cfg.search_mode} accept={cfg.accept_mode} "
          f"replicas={cfg.replicas} threads={threads} cfg_hash={cfg_hash}")
    if not cfg.seed_from_corpus:
        print("[sa814] --no-seed: ignoring data/*.txt and any prior run outputs, "
              "starting every replica from a fresh random grid")

    st = None
    if cfg.resume and not cfg.fresh:
        ck = checkpoint.load(run_dir)
        meta = checkpoint.load_meta(run_dir)
        if ck is not None:
            if meta is not None and meta.get("cfg_hash") != cfg_hash:
                print(f"[sa814] checkpoint cfg_hash mismatch "
                      f"({meta.get('cfg_hash')} != {cfg_hash}); reseeding fresh from its best grid only.")
                st = _fresh_state(cfg, run_dir, data_dir, rng)
                if ck.best_score > st["best_score"]:
                    st["best_grid"] = ck.best_grid.copy()
                    st["best_score"] = int(ck.best_score)
                    st["best_energy"] = float(ck.best_energy)
                    st["grids"][0] = ck.best_grid.copy()
                    core.build_dmask(st["grids"][0], st["dmasks"][0])
            else:
                st = _resumed_state(ck, cfg, rng)

    if st is None:
        st = _fresh_state(cfg, run_dir, data_dir, rng)

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

    def save_now():
        checkpoint.save(run_dir, _to_checkpoint_state(st), cfg.to_dict(), cfg_hash)

    # Ensure best.txt always reflects the best-known grid, even if this
    # session never beats it: a fresh run seeded from an already-strong
    # corpus grid (or a resumed run that stalls) would otherwise never write
    # best.txt at all, since that only happened on a *new* record before.
    checkpoint.write_best(run_dir, st["best_grid"], st["best_score"])

    print(f"[sa814] starting from best_score={st['best_score']} total_iters={st['total_iters']}")

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
            accept_mode, iters_per_segment, n_segments, do_swaps,
            st["accept_counter"], st["move_counter"], st["swap_accept"], st["swap_attempt"],
        )
        block_elapsed = time.perf_counter() - t_block
        iters_done = cfg.replicas * iters_per_segment * n_segments
        st["total_iters"] += iters_done
        st["elapsed_seconds"] = time.perf_counter() - wall_start

        measured_iters_per_sec = iters_done / max(block_elapsed, 1e-6)
        if block_elapsed > 0:
            target_total_iters = max(int(measured_iters_per_sec * cfg.block_seconds), cfg.replicas)
            n_segments = max(1, target_total_iters // (cfg.replicas * iters_per_segment))

        cur_best_idx = int(np.argmax(st["scores_arr"]))
        cur_best_score = int(st["scores_arr"][cur_best_idx])
        cur_best_energy = float(st["energies"][cur_best_idx])
        if (cur_best_score > st["best_score"] or
                (cur_best_score == st["best_score"] and cur_best_energy > st["best_energy"])):
            st["best_score"] = cur_best_score
            st["best_energy"] = cur_best_energy
            st["best_grid"] = st["grids"][cur_best_idx].copy()
            st["iters_since_best"] = 0
            st["stagnant_cycles"] = 0
            print(f"[sa814] NEW RECORD: score={cur_best_score} "
                  f"(iter {st['total_iters']}, t={st['elapsed_seconds']:.1f}s)")
            checkpoint.write_best(run_dir, st["best_grid"], st["best_score"])
            checkpoint.append_record(run_dir, st["best_grid"], st["best_score"],
                                      st["total_iters"], st["elapsed_seconds"])
            save_now()
            last_checkpoint = time.perf_counter()
        else:
            st["iters_since_best"] += iters_done

        if cfg.search_mode == "anneal":
            for i in range(cfg.replicas):
                st["cycle_pos"][i] += iters_done
                frac = min(1.0, st["cycle_pos"][i] / cfg.cycle_iters)
                st["temps"][i] = st["t0"] * ((st["t_end"] / st["t0"]) ** frac)

        if st["iters_since_best"] >= cfg.stagnation_iters:
            st["stagnant_cycles"] += 1
            use_best = st["stagnant_cycles"] < 3 or cfg.restart_from != "best"
            print(f"[sa814] stagnation ({st['iters_since_best']} iters without improvement) -> "
                  f"reheating (stagnant_cycles={st['stagnant_cycles']}, "
                  f"restart_from={'best' if use_best else 'current'})")
            for i in range(cfg.replicas):
                kick = 2 + (i % 6)
                _reseed_replica(st, cfg, i, from_best=use_best, kick_strength=kick)
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
            accept_rate = total_accept / max(total_moves, 1)
            print(f"[sa814] iter={st['total_iters']:>12d}  t={st['elapsed_seconds']:>7.1f}s  "
                  f"best={st['best_score']:>5d}  cur_best={cur_best_score:>5d}  "
                  f"mean_E={mean_e:>10.2f}  accept={accept_rate:>5.1%}  "
                  f"T=[{np.min(st['temps']):.4g},{np.max(st['temps']):.4g}]  "
                  f"{measured_iters_per_sec:>10.0f} it/s")
            checkpoint.append_progress(run_dir, {
                "iter": st["total_iters"], "elapsed": round(st["elapsed_seconds"], 1),
                "best_score": st["best_score"], "mean_energy": round(mean_e, 3),
                "accept_rate": round(accept_rate, 4),
                "T_min": float(np.min(st["temps"])), "T_max": float(np.max(st["temps"])),
            })
            last_print = now

        if cfg.max_seconds is not None and st["elapsed_seconds"] >= cfg.max_seconds:
            print(f"[sa814] stopping: reached max_seconds={cfg.max_seconds}")
            break
        if cfg.max_iters is not None and st["total_iters"] >= cfg.max_iters:
            print(f"[sa814] stopping: reached max_iters={cfg.max_iters}")
            break
        if cfg.target_score is not None and st["best_score"] >= cfg.target_score:
            print(f"[sa814] stopping: reached target_score={cfg.target_score}")
            break
        if stop_flag["stop"]:
            print("[sa814] stopping: signal received")
            break

    checkpoint.write_best(run_dir, st["best_grid"], st["best_score"])
    save_now()
    print(f"[sa814] final: best_score={st['best_score']} total_iters={st['total_iters']} "
          f"elapsed={st['elapsed_seconds']:.1f}s -> {run_dir / 'best.txt'}")
