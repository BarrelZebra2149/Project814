"""
Proves the new bitmask scorer (core814.is_formable / core814.evaluate) computes
exactly the same answers as the original scorer (originally duplicated
verbatim across 814_cpu_score_first.py and 814_cpu_count_first.py; consolidated
in legacy_reference.py -- see that file's docstring for why).

The original `original_has_path_fast` pushes onto a fixed 500-slot stack with
NO overflow check, so on a grid with enough digit repetition it can silently
overflow / produce wrong results. When the new scorer and the original
disagree, legacy_reference.oracle_has_path (an unbounded, overflow-free
Python-list stack, ported from code/check_grids.py's `_has_path`) is used as
the tie-breaking oracle to determine which one is actually correct.

Usage:  python verify_scorer.py [--trials N] [--seed S]
Exit code 0 = new scorer matches (or is proven correct where the original
silently failed); exit code 1 = an unexplained mismatch was found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core814 as core
import legacy_reference as ref

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def random_grid(rng):
    return rng.integers(0, 10, size=(core.ROWS, core.COLS)).astype(np.uint8)


def skewed_grid(rng, n_digits_used):
    """Grids with heavy digit repetition -- the regime most likely to trigger the
    original's unchecked 500-slot stack overflow, since repeated digits blow up
    the DFS branching factor."""
    digits = rng.choice(10, size=n_digits_used, replace=False)
    return rng.choice(digits, size=(core.ROWS, core.COLS)).astype(np.uint8)


def load_corpus_grids():
    grids = []
    if not DATA_DIR.exists():
        return grids
    for path in sorted(DATA_DIR.glob("*.txt")):
        lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if len(l.strip()) == core.COLS and l.strip().isdigit()]
        for i in range(0, len(lines) - core.ROWS + 1, core.ROWS):
            block = lines[i:i + core.ROWS]
            if len(block) != core.ROWS:
                continue
            flat = [int(ch) for row in block for ch in row]
            grids.append(np.array(flat, dtype=np.uint8).reshape(core.ROWS, core.COLS))
    return grids


def compare_is_formable(grids, numbers):
    mismatches = []
    for gi, grid in enumerate(grids):
        dmask = np.zeros((10, core.ROWS), dtype=np.int64)
        core.build_dmask(grid, dmask)
        buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)
        for n in numbers:
            digits_arr = ref.original_get_digits_math(np.int64(n))
            nd = core.fill_digits(n, buf)
            assert nd == len(digits_arr) and list(buf[:nd]) == list(digits_arr), \
                f"digit extraction mismatch for n={n}"

            orig_result = bool(ref.original_has_path_fast(grid, digits_arr))
            new_result = bool(core.is_formable(dmask, buf, nd))

            if orig_result != new_result:
                oracle_result = ref.oracle_has_path(grid, list(digits_arr))
                mismatches.append((gi, n, orig_result, new_result, oracle_result))
    return mismatches


def compare_evaluate(grids):
    mismatches = []
    for gi, grid in enumerate(grids):
        dmask = np.zeros((10, core.ROWS), dtype=np.int64)
        core.build_dmask(grid, dmask)
        stamp = np.full(core.UPPER, -1, dtype=np.int64)
        buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)

        new_score, new_look, new_count = core.evaluate(
            dmask, stamp, 1, look_window=0, count_lo=1000, count_hi=10000,
            want_count=True, digit_buf=buf,
        )

        orig_score, orig_formable = ref.original_evaluate(grid)

        if new_score != orig_score or new_count != orig_formable:
            mismatches.append((gi, orig_score, new_score, orig_formable, new_count))
    return mismatches


def compare_against_oracle(grids):
    """Independent third confirmation: compares the new scorer directly against
    legacy_reference.oracle_evaluate, which involves no numba original code at
    all (pure Python, unbounded). Slow, so only run on a handful of grids."""
    mismatches = []
    for gi, grid in enumerate(grids):
        dmask = np.zeros((10, core.ROWS), dtype=np.int64)
        core.build_dmask(grid, dmask)
        stamp = np.full(core.UPPER, -1, dtype=np.int64)
        buf = np.zeros(core.DIGIT_BUF_LEN, dtype=np.int64)

        new_score, new_look, new_count = core.evaluate(
            dmask, stamp, 1, look_window=0, count_lo=1000, count_hi=10000,
            want_count=True, digit_buf=buf,
        )
        oracle_score, oracle_formable = ref.oracle_evaluate(grid)

        if new_score != oracle_score or new_count != oracle_formable:
            mismatches.append((gi, oracle_score, new_score, oracle_formable, new_count))
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--numbers-per-grid", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[1/4] is_formable equivalence: {args.trials} random grids x "
          f"{args.numbers_per_grid} random numbers ...")
    grids = [random_grid(rng) for _ in range(args.trials // 2)]
    grids += [skewed_grid(rng, rng.integers(2, 5)) for _ in range(args.trials // 2)]
    numbers = rng.integers(1, core.UPPER, size=args.numbers_per_grid).tolist()
    mism = compare_is_formable(grids, numbers)

    unexplained = []
    original_was_wrong = 0
    for gi, n, orig_r, new_r, oracle_r in mism:
        if new_r == oracle_r and orig_r != oracle_r:
            original_was_wrong += 1
        else:
            unexplained.append((gi, n, orig_r, new_r, oracle_r))

    print(f"  {len(mism)} disagreement(s) between original and new scorer.")
    if mism:
        print(f"  -> {original_was_wrong} were the original's known stack-overflow bug "
              f"(new scorer agreed with the unbounded oracle).")
    if unexplained:
        print(f"  !! {len(unexplained)} UNEXPLAINED mismatch(es) (new scorer disagreed with oracle):")
        for row in unexplained[:20]:
            print(f"     grid#{row[0]} n={row[1]} orig={row[2]} new={row[3]} oracle={row[4]}")

    print(f"\n[2/4] evaluate() equivalence (score + formable count) on "
          f"{len(grids)} synthetic grids ...")
    mism2 = compare_evaluate(grids)
    print(f"  {len(mism2)} disagreement(s).")
    for row in mism2[:20]:
        print(f"     grid#{row[0]} orig_score={row[1]} new_score={row[2]} "
              f"orig_formable={row[3]} new_count={row[4]}")

    print("\n[3/4] evaluate() equivalence on the real data/*.txt corpus ...")
    corpus = load_corpus_grids()
    print(f"  Loaded {len(corpus)} grids from data/*.txt")
    mism3 = compare_evaluate(corpus) if corpus else []
    print(f"  {len(mism3)} disagreement(s).")
    for row in mism3[:20]:
        print(f"     grid#{row[0]} orig_score={row[1]} new_score={row[2]} "
              f"orig_formable={row[3]} new_count={row[4]}")

    print("\n[4/4] independent oracle check (pure Python, no numba original code) "
          "on a handful of grids ...")
    oracle_sample = grids[:3] + (corpus[:3] if corpus else [])
    mism4 = compare_against_oracle(oracle_sample)
    print(f"  {len(mism4)} disagreement(s) out of {len(oracle_sample)} grids checked.")
    for row in mism4[:20]:
        print(f"     grid#{row[0]} oracle_score={row[1]} new_score={row[2]} "
              f"oracle_formable={row[3]} new_count={row[4]}")

    ok = not unexplained and not mism2 and not mism3 and not mism4
    print("\n" + ("PASS: new scorer is semantically equivalent to the original "
                  "(and correct where the original silently overflowed)." if ok
                  else "FAIL: unexplained discrepancies found -- see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
