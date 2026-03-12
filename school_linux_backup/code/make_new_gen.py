import random
import numpy as np
import os
from numba import njit

# ====================== CONFIGURATION ======================
INPUT_FILE = 'final_gen2.txt'
OUTPUT_FILE = 'look.txt'

GRID_ROWS = 8
GRID_COLS = 14

# How many times to copy each good grid
COPIES_PER_GRID = 1

# ====================== EVALUATION ======================
def eval_814_heuristic(individual):
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

    return float(current_score)

@njit
def _has_path(grid: np.ndarray, digits: np.ndarray) -> bool:
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
    if n <= 0:
        return 0
    digits_list = [int(d) for d in str(n)]
    digits = np.array(digits_list, dtype=np.int64)
    return 1 if _has_path(grid, digits) else 0


# ====================== MAIN ======================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    grids = []
    block = []

    print(f"Reading from '{INPUT_FILE}'...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == GRID_COLS and line.isdigit():
                block.append(line)
                if len(block) == GRID_ROWS:
                    flat = [int(d) for row in block for d in row]
                    grids.append(flat)
                    block = []

    print(f"Loaded {len(grids)} grids.")

    # Filter 1000 ~ 2500 range
    good_grids = []
    for genome in grids:
        score = eval_814_heuristic(genome)
        if 4883 <= score <= 4883:
            good_grids.append(genome)

    # Remove duplicates using set
    unique_good = []
    seen = set()
    for g in good_grids:
        t = tuple(g)
        if t not in seen:
            seen.add(t)
            unique_good.append(g)

    print(f"Found {len(unique_good)} unique grids in 3500~5000 range.")

    # Generate new file by copying (no mutation, no crossover)
    print(f"Creating new file with {len(unique_good) * COPIES_PER_GRID} grids...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for grid in unique_good:
            for _ in range(COPIES_PER_GRID):         
                grid_2d = np.array(grid).reshape(GRID_ROWS, GRID_COLS)
                for row in grid_2d:
                    f.write(''.join(map(str, row)) + '\n')
                f.write('\n')                         

    print(f"Done! Saved to '{OUTPUT_FILE}'")
    print(f"Total grids created: {len(unique_good) * COPIES_PER_GRID}")

if __name__ == "__main__":
    main()
