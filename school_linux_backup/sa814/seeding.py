"""
Seeding: builds the initial population of replica grids from the existing
corpus (code/data/*.txt, plus any previous sa814 run's best.txt / records.txt)
instead of throwing that history away. Falls back to random grids to fill out
any remaining replica slots.

Parsing rule matches the original `load_previous_best` in 814_cpu_*.py and
code/check_grids.py: any run of exactly ROWS consecutive lines that are each
COLS digit characters is one grid; blank lines / comments / anything else
just breaks the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

import core814 as core


def parse_grids_from_text(text: str) -> List[np.ndarray]:
    lines = text.splitlines()
    grids = []
    block: List[str] = []
    for raw in lines:
        line = raw.strip()
        if len(line) == core.COLS and line.isdigit():
            block.append(line)
            if len(block) == core.ROWS:
                flat = [int(ch) for row in block for ch in row]
                grids.append(np.array(flat, dtype=np.uint8).reshape(core.ROWS, core.COLS))
                block = []
        else:
            block = []
    return grids


def parse_grids_from_file(path: Path) -> List[np.ndarray]:
    if not path.exists():
        return []
    return parse_grids_from_text(path.read_text(encoding="utf-8", errors="ignore"))


def score_grid(grid: np.ndarray) -> int:
    dmask = np.zeros((10, core.ROWS), dtype=np.int64)
    core.build_dmask(grid.astype(np.uint8), dmask)
    stamp = np.full(core.UPPER, -1, dtype=np.int64)
    buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
    score, _look, _count = core.evaluate(dmask, stamp, 1, 0, 1000, 10000, False, buf)
    return int(score)


def load_corpus(data_dir: Path, extra_seed_file: str | None, run_dir: Path | None) -> List[np.ndarray]:
    """Collects candidate grids from every known source, highest priority first:
    a previous sa814 run's best.txt/records.txt for this run_name, an optional
    user-supplied --seed-file, and the original data/*.txt corpus."""
    grids: List[np.ndarray] = []

    if run_dir is not None:
        grids += parse_grids_from_file(run_dir / "best.txt")
        grids += parse_grids_from_file(run_dir / "records.txt")

    if extra_seed_file:
        grids += parse_grids_from_file(Path(extra_seed_file))

    if data_dir.exists():
        for path in sorted(data_dir.glob("*.txt")):
            grids += parse_grids_from_file(path)

    return grids


def build_initial_replicas(replicas: int, data_dir: Path, extra_seed_file: str | None,
                            run_dir: Path | None, rng: np.random.Generator) -> np.ndarray:
    """Returns a uint8[replicas, ROWS, COLS] array: the best-scoring corpus grids
    (deduplicated), ranked, filling remaining slots with fresh random grids."""
    corpus = load_corpus(data_dir, extra_seed_file, run_dir)

    seen = set()
    unique = []
    for g in corpus:
        key = g.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(g)

    scored = sorted(unique, key=score_grid, reverse=True)

    out = np.empty((replicas, core.ROWS, core.COLS), dtype=np.uint8)
    n_from_corpus = min(len(scored), replicas)
    for i in range(n_from_corpus):
        out[i] = scored[i]
    for i in range(n_from_corpus, replicas):
        out[i] = rng.integers(0, 10, size=(core.ROWS, core.COLS)).astype(np.uint8)

    return out
