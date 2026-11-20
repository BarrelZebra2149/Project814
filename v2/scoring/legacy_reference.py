"""
Consolidated reference/oracle scoring functions, for testing core814.py only.

Before this file existed, these were scattered: the safe unbounded path check
lived only inside code/check_grids.py, and the original (buggy-stack) scorer
had to be pulled in via importlib from code/814_cpu_count_first.py -- which
executes `from deap import base, creator, tools` at module level just to reach
two small numba functions, so verifying the new scorer required a `deap`
install it doesn't otherwise need. Everything here is copied, not
reimplemented, so it stays a trustworthy, independent ground truth.

Two layers, most to least trustworthy:

  1. oracle_*        Pure Python, unbounded, no numba, no fixed-size buffers.
                      Ported from code/check_grids.py's `_has_path`, minus its
                      @njit decorator: modern numba (this repo runs 0.63.1)
                      dropped reflected-list support, so a python-list stack
                      under @njit no longer compiles anyway, and a slow but
                      unimpeachable oracle is exactly what a tie-breaker needs
                      to be. Use these whenever anything else disagrees.

  2. original_*       Numba-jitted, byte-for-byte copies of the pre-existing
                      `_has_path_fast` / `_get_digits_math` / `_reverse_int_math`
                      / `_evaluate_core_numba` (identical in both
                      814_cpu_score_first.py and 814_cpu_count_first.py). Kept
                      for exact behavioral comparison, INCLUDING the fixed
                      500-slot stack with no overflow check -- do not "fix"
                      that here; the whole point is to reproduce the original
                      exactly, bug and all, so tests can tell the two apart.

score_grid_oracle() ties oracle_has_path + oracle_get_digits/oracle_reverse_int
together into the same (score, formable) shape core814.evaluate() returns, so
tests can compare like for like without touching numba at all.
"""

from __future__ import annotations

import numpy as np
from numba import njit

UPPER = 50000  # matches core814.UPPER / the original _evaluate_core_numba's scan bound


# ===========================================================================
# 1. Oracle: pure Python, unbounded, no fixed-size stack.
#    Ported from code/check_grids.py's `_has_path` (list-based stack there
#    too; only the @njit decorator is dropped, for the reason above).
# ===========================================================================

_DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def oracle_has_path(grid: np.ndarray, digits) -> bool:
    """Walk-existence check (cells may be revisited), no bound on search size."""
    rows, cols = grid.shape
    digit_len = len(digits)
    if digit_len == 0:
        return False
    stack = []
    first_digit = digits[0]
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == first_digit:
                stack.append((r, c, 0))
    while stack:
        r, c, idx = stack.pop()
        if idx == digit_len - 1:
            return True
        next_idx = idx + 1
        next_digit = digits[next_idx]
        for dr, dc in _DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == next_digit:
                stack.append((nr, nc, next_idx))
    return False


def oracle_get_digits(n: int):
    """Most-significant-digit-first digit list of n (n >= 1)."""
    return [int(ch) for ch in str(n)]


def oracle_reverse_int(n: int) -> int:
    return int(str(n)[::-1])


def oracle_evaluate(grid: np.ndarray, count_lo: int = 1000, count_hi: int = 10000):
    """Pure-Python ground truth: (score, formable_count). No numba, no shared
    stamp/gen cache tricks, just a straightforward found-set scan. Slow --
    use only for tests, not inside any hot loop."""
    found = set()
    n = 1
    current_score = UPPER - 1
    while n < UPPER:
        rev_n = oracle_reverse_int(n)
        if n in found or (n % 10 != 0 and rev_n < UPPER and rev_n in found):
            n += 1
            continue
        if oracle_has_path(grid, oracle_get_digits(n)):
            found.add(n)
            if rev_n < UPPER:
                found.add(rev_n)
        else:
            current_score = n - 1
            break
        n += 1

    score = current_score
    formable = max(0, min(count_hi - 1, score) - count_lo + 1)
    start = max(count_lo, score + 1)
    for num in range(start, count_hi):
        rev_num = oracle_reverse_int(num)
        if num in found or (num % 10 != 0 and rev_num < UPPER and rev_num in found):
            formable += 1
            continue
        if oracle_has_path(grid, oracle_get_digits(num)):
            formable += 1
            found.add(num)
            if rev_num < UPPER:
                found.add(rev_num)

    return score, formable


# ===========================================================================
# 2. Original scorer: numba-jitted, unchanged from 814_cpu_count_first.py /
#    814_cpu_score_first.py (both define the identical four functions below;
#    score_first just never calls _evaluate_core_numba). Copied verbatim,
#    including the 500-slot stack with no overflow guard -- kept intentionally
#    unfixed so tests can detect exactly when/whether that bug fires.
# ===========================================================================

@njit(fastmath=True)
def original_has_path_fast(grid, digits):
    rows, cols = grid.shape
    digit_len = digits.shape[0]
    deltas = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    stack = np.empty((500, 3), dtype=np.int64)
    head = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == digits[0]:
                if digit_len == 1:
                    return True
                stack[head, 0], stack[head, 1], stack[head, 2] = r, c, 0
                head += 1
    while head > 0:
        head -= 1
        r, c, idx = stack[head, 0], stack[head, 1], stack[head, 2]
        next_digit = digits[idx + 1]
        for i in range(8):
            nr = r + deltas[i, 0]
            nc = c + deltas[i, 1]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == next_digit:
                if idx + 1 == digit_len - 1:
                    return True
                stack[head, 0], stack[head, 1], stack[head, 2] = nr, nc, idx + 1
                head += 1
    return False


@njit(fastmath=True)
def original_get_digits_math(n):
    temp = np.empty(6, dtype=np.int64)
    idx = 0
    while n > 0:
        temp[idx] = n % 10
        n //= 10
        idx += 1
    result = np.empty(idx, dtype=np.int64)
    for i in range(idx):
        result[i] = temp[idx - 1 - i]
    return result


@njit(fastmath=True)
def original_reverse_int_math(n):
    rev = 0
    while n > 0:
        rev = rev * 10 + (n % 10)
        n //= 10
    return rev


@njit
def original_evaluate_core_numba(grid_1d, rows, cols):
    """Returns (formable, current_score) -- yes, that order; matches the
    original's own (misleading) docstring + count_first's unpack/re-swap."""
    grid = grid_1d.reshape((rows, cols))
    found = np.zeros(50000, dtype=np.bool_)
    current_score, n = 49999, 1
    while n < 50000:
        rev_n = original_reverse_int_math(n)
        if found[n] or (n % 10 != 0 and rev_n < 50000 and found[rev_n]):
            n += 1
            continue
        digits = original_get_digits_math(n)
        if original_has_path_fast(grid, digits):
            found[n] = True
            if rev_n < 50000:
                found[rev_n] = True
        else:
            current_score = n - 1
            break
        n += 1
    formable = max(0, min(10000, current_score) - 1000 + 1)
    for num in range(max(1000, current_score + 1), 10000):
        rev_num = original_reverse_int_math(num)
        if found[num] or (num % 10 != 0 and rev_num < 50000 and found[rev_num]):
            formable += 1
            continue
        digits = original_get_digits_math(num)
        if original_has_path_fast(grid, digits):
            formable += 1
            found[num] = True
            if rev_num < 50000:
                found[rev_num] = True
    return float(formable), float(current_score)


def original_evaluate(grid: np.ndarray):
    """Convenience wrapper matching oracle_evaluate's (score, formable) shape
    (the raw numba function returns (formable, score) -- see its docstring)."""
    grid_1d = grid.astype(np.int64).reshape(-1)
    formable, score = original_evaluate_core_numba(grid_1d, grid.shape[0], grid.shape[1])
    return int(score), int(formable)
