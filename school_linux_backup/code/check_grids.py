import numpy as np
from numba import njit
import os
import sys

# ====================== CONFIGURATION ======================
INPUT_FILE = 'final_gen2.txt'
GRID_ROWS = 8
GRID_COLS = 14
GRID_SIZE = GRID_ROWS * GRID_COLS


# ====================== EVALUATION FUNCTIONS ======================
@njit
def _has_path(grid: np.ndarray, digits: np.ndarray) -> bool:
    """Check if a path exists for the digit sequence (8-directional movement)"""
    rows, cols = grid.shape
    digit_len = digits.shape[0]
    if digit_len == 0:
        return False
    deltas = np.array([[-1, -1], [-1, 0], [-1, 1],
                       [0, -1], [0, 1],
                       [1, -1], [1, 0], [1, 1]], dtype=np.int64)
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
        if next_idx < digit_len:
            next_digit = digits[next_idx]
            for i in range(8):
                nr = r + deltas[i, 0]
                nc = c + deltas[i, 1]
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == next_digit:
                    stack.append((nr, nc, next_idx))
    return False


def count_occurrences(grid, n):
    """Count whether a number can be formed as a path in the grid"""
    if n <= 0:
        return 0
    digits_list = [int(d) for d in str(n)]
    digits = np.array(digits_list, dtype=np.int64)
    return 1 if _has_path(grid, digits) else 0


def eval_814_heuristic(individual):
    """Evaluate the grid: return consecutive score and formable count"""
    grid = np.array(individual, dtype=np.int64).reshape(GRID_ROWS, GRID_COLS)
    MAX_N = 50000
    found_set = set()
    n = 1
    while n < MAX_N:
        rev_str = str(n)[::-1]
        rev_n = int(rev_str)
        if n in found_set or (n % 10 != 0 and rev_n in found_set):
            n += 1
            continue
        if count_occurrences(grid, n):
            found_set.add(n)
            found_set.add(rev_n)
        else:
            current_score = n - 1
            break
        n += 1
    else:
        current_score = MAX_N - 1

    formable_count = max(0, min(9999, current_score) - 1000 + 1)
    start_num = max(1000, current_score + 1)
    for num in range(start_num, 10000):
        rev_str = str(num)[::-1]
        rev_num = int(rev_str)
        if num in found_set or (num % 10 != 0 and rev_num in found_set):
            formable_count += 1
            continue
        if count_occurrences(grid, num):
            formable_count += 1
            found_set.add(num)
            found_set.add(rev_num)

    return float(current_score), float(formable_count)


# ====================== MAIN ======================
def main():
    """Main function: Load unique grids, evaluate them, and show top results by consecutive score"""
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' does not exist.")
        return

    grids = []
    block = []

    print(f"Reading unique grids from '{INPUT_FILE}'...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == GRID_COLS and line.isdigit():
                block.append(line)
                if len(block) == GRID_ROWS:
                    # Convert to flat list (for evaluation)
                    flat = [int(d) for row in block for d in row]
                    grids.append(flat)
                    block = []

    print(f"Loaded {len(grids)} unique grids.\n")
    print("Evaluating all grids... (this may take some time)\n")

    # Evaluate every grid
    results = []
    for i, genome in enumerate(grids):
        consecutive, density = eval_814_heuristic(genome)
        results.append((consecutive, density, i + 1))  # (consecutive, density, original index)
        
        if (i + 1) % 50 == 0 or i == len(grids) - 1:
            print(f"Evaluated {i + 1}/{len(grids)} grids...")

    # Sort by consecutive score descending (like selBest)
    results.sort(key=lambda x: x[0], reverse=True)

    # Display top 50
    print("\n" + "=" * 80)
    print("SELBEST - TOP GRIDS SORTED BY CONSECUTIVE SCORE")
    print("=" * 80)

    for rank, (consecutive, density, original_idx) in enumerate(results[:100], 1):
        print(f"Rank {rank:2d} | Consecutive: {consecutive:6.0f} | "
              f"Density: {density:6.0f} | Original Unique Rank: {original_idx}")

    print("\n" + "=" * 80)
    print(f"Total unique grids evaluated: {len(results)}")
    print(f"Best score: {results[0][0]:.0f} | Worst in top 100: {results[99][0]:.0f}")
    print("=" * 80)


if __name__ == "__main__":
    main()