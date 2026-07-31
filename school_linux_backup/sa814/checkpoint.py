"""
Checkpointing for sa814 runs: atomic mid-run state saves + resume, plus the
human-readable output files (best grid, new-record history, progress log).

This is the piece the original 814_cpu_*.py scripts were missing entirely:
they only ever appended the best grid to a seed file on a new record, so
killing the process lost all temperature / RNG / iteration-count state. Here,
every checkpoint captures enough to resume an SA/PT run exactly where it left
off (replica grids, energies, temperatures, RNG states, LAHC/DLAS history,
counters, elapsed time).

Layout under runs/<run_name>/:
    checkpoint.npz        atomically-written numeric state (see save())
    checkpoint.prev.npz   previous checkpoint, kept as a fallback
    meta.json             human-readable summary + cfg_hash + format_version
    best.txt              current best grid, 8 lines x 14 digits (submission format)
    records.txt           append-only new-record history, same 8x14 block format
                           as the original data/*.txt corpus (+ a comment line),
                           so code/check_grids.py, make_new_gen.py, permutation.py
                           can still read it unmodified.
    progress.csv          iter,elapsed,best_score,mean_energy,accept_rate,T_min,T_max
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

FORMAT_VERSION = 1


@dataclass
class CheckpointState:
    grids: np.ndarray            # uint8 [R, 8, 14]
    energies: np.ndarray          # f8 [R]
    scores: np.ndarray            # i4 [R]
    looks: np.ndarray             # i4 [R]
    counts: np.ndarray            # i4 [R]
    temps: np.ndarray             # f8 [R]
    rng_states: np.ndarray        # u8 [R, 2]
    hist: np.ndarray              # f8 [R, lahc_len]
    lahc_pos: np.ndarray          # i8 [R]
    cycle_pos: np.ndarray         # i8 [R]  (iterations into the current cooling cycle, anneal mode)
    accept_counter: np.ndarray    # i8 [R]
    move_counter: np.ndarray      # i8 [R, 5]
    swap_accept: np.ndarray       # i8 [R-1]
    swap_attempt: np.ndarray      # i8 [R-1]
    best_grid: np.ndarray         # uint8 [8, 14]
    best_score: int
    best_energy: float
    total_iters: int
    iters_since_best: int
    elapsed_seconds: float
    stagnant_cycles: int = 0


def run_root(base_dir: Path, run_name: str) -> Path:
    d = base_dir / "runs" / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_bytes(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    if path.exists():
        prev = path.with_name(path.stem + ".prev" + path.suffix)
        os.replace(path, prev)
    os.replace(tmp, path)


def save(run_dir: Path, state: CheckpointState, cfg_dict: dict, cfg_hash: str) -> None:
    npz_path = run_dir / "checkpoint.npz"

    # np.savez insists on adding ".npz" to str/Path targets that don't already
    # end with it, which corrupts our ".npz.tmp" naming. Write to an explicit
    # open file object instead, which savez respects literally.
    tmp_path = npz_path.with_name(npz_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(
            f,
            grids=state.grids,
            energies=state.energies,
            scores=state.scores,
            looks=state.looks,
            counts=state.counts,
            temps=state.temps,
            rng_states=state.rng_states,
            hist=state.hist,
            lahc_pos=state.lahc_pos,
            cycle_pos=state.cycle_pos,
            accept_counter=state.accept_counter,
            move_counter=state.move_counter,
            swap_accept=state.swap_accept,
            swap_attempt=state.swap_attempt,
            best_grid=state.best_grid,
            best_score=np.int64(state.best_score),
            best_energy=np.float64(state.best_energy),
            total_iters=np.int64(state.total_iters),
            iters_since_best=np.int64(state.iters_since_best),
            elapsed_seconds=np.float64(state.elapsed_seconds),
            stagnant_cycles=np.int64(state.stagnant_cycles),
            format_version=np.int64(FORMAT_VERSION),
        )
        f.flush()
        os.fsync(f.fileno())

    if npz_path.exists():
        prev_path = run_dir / "checkpoint.prev.npz"
        os.replace(npz_path, prev_path)
    os.replace(tmp_path, npz_path)

    meta = {
        "format_version": FORMAT_VERSION,
        "cfg_hash": cfg_hash,
        "cfg": cfg_dict,
        "best_score": int(state.best_score),
        "best_energy": float(state.best_energy),
        "total_iters": int(state.total_iters),
        "iters_since_best": int(state.iters_since_best),
        "elapsed_seconds": float(state.elapsed_seconds),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "replicas": int(state.grids.shape[0]),
    }
    meta_path = run_dir / "meta.json"
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_meta, meta_path)


def _load_npz(path: Path) -> Optional[CheckpointState]:
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return CheckpointState(
                grids=z["grids"],
                energies=z["energies"],
                scores=z["scores"],
                looks=z["looks"],
                counts=z["counts"],
                temps=z["temps"],
                rng_states=z["rng_states"],
                hist=z["hist"],
                lahc_pos=z["lahc_pos"],
                cycle_pos=z["cycle_pos"],
                accept_counter=z["accept_counter"],
                move_counter=z["move_counter"],
                swap_accept=z["swap_accept"],
                swap_attempt=z["swap_attempt"],
                best_grid=z["best_grid"],
                best_score=int(z["best_score"]),
                best_energy=float(z["best_energy"]),
                total_iters=int(z["total_iters"]),
                iters_since_best=int(z["iters_since_best"]),
                elapsed_seconds=float(z["elapsed_seconds"]),
                stagnant_cycles=int(z["stagnant_cycles"]) if "stagnant_cycles" in z else 0,
            )
    except Exception as exc:  # noqa: BLE001 - corrupt/partial file, fall back
        print(f"[checkpoint] failed to load {path}: {exc!r}")
        return None


def load(run_dir: Path) -> Optional[CheckpointState]:
    state = _load_npz(run_dir / "checkpoint.npz")
    if state is not None:
        return state
    print("[checkpoint] checkpoint.npz missing/corrupt, trying checkpoint.prev.npz ...")
    return _load_npz(run_dir / "checkpoint.prev.npz")


def load_meta(run_dir: Path) -> Optional[dict]:
    path = run_dir / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _grid_to_lines(grid: np.ndarray) -> list:
    return ["".join(str(int(v)) for v in row) for row in grid]


def write_best(run_dir: Path, grid: np.ndarray, score: int) -> None:
    path = run_dir / "best.txt"
    tmp = path.with_name(path.name + ".tmp")
    lines = _grid_to_lines(grid)
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_record(run_dir: Path, grid: np.ndarray, score: int, total_iters: int, elapsed: float) -> None:
    path = run_dir / "records.txt"
    lines = _grid_to_lines(grid)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"# score={score} iters={total_iters} t={elapsed:.1f}s\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


def append_progress(run_dir: Path, row: dict) -> None:
    path = run_dir / "progress.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
