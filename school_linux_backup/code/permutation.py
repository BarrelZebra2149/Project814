import numpy as np
import itertools
from numba import njit
from tqdm import tqdm
import os

# =============================================================================
# 1. File I/O Logic
# =============================================================================
def load_all_grids(filename):
    """Extracts all 8x14 grids from the specified file and returns them as a list."""
    grids = []
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return grids

    with open(filename, 'r') as f:
        current_grid = []
        for line in f:
            line = line.strip()
            # Check if the line is a 14-digit string
            if len(line) == 14 and line.isdigit():
                current_grid.append([int(d) for d in line])
                if len(current_grid) == 8:
                    grids.append(np.array(current_grid, dtype=np.int64).flatten())
                    current_grid = []
    return grids


# =============================================================================
# 2. High-Performance Mapping & Evaluation Engine (Numba)
# =============================================================================
@njit(fastmath=True)
def _has_path_fast(grid, digits):
    """Ultra-fast pathfinding using a static stack."""
    rows, cols = grid.shape
    digit_len = digits.shape[0]
    deltas = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    stack = np.empty((500, 3), dtype=np.int64)
    head = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == digits[0]:
                if digit_len == 1: return True
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
                if idx + 1 == digit_len - 1: return True
                stack[head, 0], stack[head, 1], stack[head, 2] = nr, nc, idx + 1
                head += 1
    return False

@njit(fastmath=True)
def _get_digits_math(n):
    """Extract digits from an integer using mathematical operations."""
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
def _reverse_int_math(n):
    """Reverse an integer using mathematical operations."""
    rev = 0
    while n > 0:
        rev = rev * 10 + (n % 10)
        n //= 10
    return rev

@njit
def _evaluate_core_numba(grid_1d, rows, cols):
    """Core evaluation logic optimized with Numba."""
    grid = grid_1d.reshape((rows, cols))
    found = np.zeros(50000, dtype=np.bool_)
    current_score, n = 49999, 1
    while n < 50000:
        rev_n = _reverse_int_math(n)
        if found[n] or (n % 10 != 0 and rev_n < 50000 and found[rev_n]):
            n += 1
            continue
        digits = _get_digits_math(n)
        if _has_path_fast(grid, digits):
            found[n] = True
            if rev_n < 50000: found[rev_n] = True
        else:
            current_score = n - 1
            break
        n += 1
    
    formable = max(0, min(30000, current_score) - 1000 + 1)
    for num in range(max(1000, current_score + 1), 30000):
        rev_num = _reverse_int_math(num)
        if found[num] or (num % 10 != 0 and rev_num < 50000 and found[rev_num]):
            formable += 1
            continue
        digits = _get_digits_math(num)
        if _has_path_fast(grid, digits):
            formable += 1
            found[num] = True
            if rev_num < 50000: found[rev_num] = True
    return float(current_score), float(formable)

@njit(fastmath=True)
def evaluate_mapping(grid_1d, p_array):
    """Applies the given digit permutation array and evaluates the score immediately."""
    mapped = np.empty_like(grid_1d)
    for i in range(len(grid_1d)):
        mapped[i] = p_array[grid_1d[i]]

    score, formable = _evaluate_core_numba(mapped, 8, 14)
    return score, formable, mapped


# =============================================================================
# 3. Main Processor
# =============================================================================
def run_brute_force_optimization(input_file, output_file):
    grids = load_all_grids(input_file)
    print(f"Total grids found: {len(grids)}. Starting brute-force permutation search.")

    digits = np.arange(10, dtype=np.int64)
    all_perms = list(itertools.permutations(digits))  

    with open(output_file, 'w') as f_out:
        for idx, grid in enumerate(grids):
            print(f"\n[Grid {idx + 1}/{len(grids)}] Searching for optimal mapping...")

            best_score = -1.0
            best_mapped_grid = None
            best_mapping = None

            with tqdm(all_perms, desc=f"[Best 0]", leave=True, ncols=None) as pbar:
                for j, p in enumerate(pbar):
                    p_array = np.array(p, dtype=np.int64)
                    score, formable, mapped = evaluate_mapping(grid, p_array)
                    
                    if j % 2500 == 0:
                        pbar.write(f"[Set {j:7d}] | CurrScore : {score} | CurrMax: {best_score:.0f}")

                    if score > best_score:
                        best_score = score
                        best_mapped_grid = mapped
                        best_mapping = p_array
                        pbar.set_description(f"[Best {best_score:.0f}]")
                        
                        f_out.write(f"Grid {idx + 1} | New Best: {best_score:.0f} | Mapping: {list(best_mapping)}\n")
                        
                        reshaped_grid = best_mapped_grid.reshape(8, 14)
                        for row in reshaped_grid:
                            line = "".join(map(str, row))
                            f_out.write(line + "\n")
                        
                        f_out.write("\n") 
                        f_out.flush() 
            
            f_out.write(f"--- Grid {idx + 1} Completed (Final Best: {best_score:.0f}) ---\n\n")
            f_out.flush()

if __name__ == "__main__":
    # Ensure input file exists before running
    input_filename = 'target.txt'
    output_filename = 'optimized_results.txt'
    
    run_brute_force_optimization(input_filename, output_filename)