import random
import numpy as np
import sys
import os
import multiprocessing
import subprocess
import time
from deap import base, creator, tools
from numba import njit
from tqdm import tqdm
import shutil

# =============================================================================
# 1. Hyperparameters & Global Constants
# =============================================================================
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = '../data/NG_LONG.txt'
TARGET_POP = 2500
NGEN = 100000000
G_PRINT_GROUP = 1000
G_SEED_GROUP = 10000
STAGNATION_LIMIT = 100000
ELITE_SIZE = 25
ELITE_BEST_SIZE = 25 

CURRENT_CXPB = 0.5
CURRENT_MUTPB = 0.35

GLOBAL_MAX_SCORE = 0.0
STAGNATION_MODE = False
CPP_DLAS_BINARY = "./my_dlas"

TQDM_FILE = sys.stdout

# =============================================================================
# 2. DEAP Framework Setup
# =============================================================================
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def get_secure_rng():
    seed_128 = int.from_bytes(os.urandom(16), byteorder='big')
    return np.random.default_rng(seed_128)

rng = get_secure_rng()

# =============================================================================
# 3. Core Evaluation (Numba Optimized)
# =============================================================================
@njit(fastmath=True)
def _has_path_fast(grid, digits):
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
    rev = 0
    while n > 0:
        rev = rev * 10 + (n % 10)
        n //= 10
    return rev

@njit
def _evaluate_core_numba(grid_1d, rows, cols):
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
    """
    formable = max(0, min(10000, current_score) - 1000 + 1)
    for num in range(max(1000, current_score + 1), 10000):
        rev_num = _reverse_int_math(num)
        if found[num] or (num % 10 != 0 and rev_num < 50000 and found[rev_num]):
            formable += 1
            continue
        digits = _get_digits_math(num)
        if _has_path_fast(grid, digits):
            formable += 1
            found[num] = True
            if rev_num < 50000: found[rev_num] = True
    """
    return float(current_score), 0.0

import numpy as np

fitness_cache = {}

def calculate_advanced_fitness(grid, current_score):
    rows = len(grid)
    cols = len(grid[0])
    chains = []

    visited_bs = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited_bs:
                val = grid[r][c]
                cells = []
                cr, cc = r, c
                while cr < rows and cc < cols and grid[cr][cc] == val:
                    cells.append((cr, cc))
                    visited_bs.add((cr, cc))
                    cr += 1; cc += 1
                if len(cells) >= 2:
                    chains.append({'val': val, 'cells': cells})

    visited_fs = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited_fs:
                val = grid[r][c]
                cells = []
                cr, cc = r, c
                while cr < rows and cc >= 0 and grid[cr][cc] == val:
                    cells.append((cr, cc))
                    visited_fs.add((cr, cc))
                    cr += 1; cc -= 1
                if len(cells) >= 2:
                    chains.append({'val': val, 'cells': cells})

    num_chains = len(chains)
    adj = {i: set() for i in range(num_chains)}

    cell_to_chains = [[[] for _ in range(cols)] for _ in range(rows)]
    for i, chain in enumerate(chains):
        for r, c in chain['cells']:
            cell_to_chains[r][c].append(i)

    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    for i, chain in enumerate(chains):
        val_i = chain['val']
        for r, c in chain['cells']:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    for j in cell_to_chains[nr][nc]:
                        if i != j and val_i != chains[j]['val']:
                            adj[i].add(j)
                            adj[j].add(i)

    chain_bonus = 0
    visited_chains = set()
    for i in range(num_chains):
        if i not in visited_chains:
            stack = [i]
            visited_chains.add(i)
            cluster_size = 0
            while stack:
                curr = stack.pop()
                cluster_size += 1
                for nxt in adj[curr]:
                    if nxt not in visited_chains:
                        visited_chains.add(nxt)
                        stack.append(nxt)
            
            if cluster_size == 1: chain_bonus += 10
            elif cluster_size == 2: chain_bonus += 30
            elif cluster_size == 3: chain_bonus += 60  
            elif cluster_size == 4: chain_bonus += 5   
            else: chain_bonus += 0                    

    counts = np.bincount(grid.flatten(), minlength=10)
    variance_penalty = np.sum((counts - 11) ** 2)

    WEIGHT_VARIANCE = 1.5
    final_potential = current_score + chain_bonus - (WEIGHT_VARIANCE * variance_penalty)
    
    return final_potential

def eval_814_heuristic(individual):
    grid = np.array(individual).reshape(IND_ROWS, IND_COLS)
    current_score = 0 
    final_score = calculate_advanced_fitness(grid, current_score)
    return (final_score, )

def smart_evaluate(individual):
    ind_tuple = tuple(individual)
    if ind_tuple in fitness_cache:
        return fitness_cache[ind_tuple]
    fit = eval_814_heuristic(individual)
    fitness_cache[ind_tuple] = fit
    return fit

toolbox.register("evaluate", smart_evaluate)

# =============================================================================
# 4. Custom Genetic Operators
# =============================================================================
def custom_mate(ind1, ind2):
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)
    sy = rng.integers(0, IND_ROWS - 1)
    sx = rng.integers(0, IND_COLS - 1)
    ey = rng.integers(sy + 1, IND_ROWS + 1)
    ex = rng.integers(sx + 1, IND_COLS + 1)
    grid1[sy:ey, sx:ex], grid2[sy:ey, sx:ex] = grid2[sy:ey, sx:ex].copy(), grid1[sy:ey, sx:ex].copy()
    ind1[:], ind2[:] = grid1.flatten().tolist(), grid2.flatten().tolist()
    return ind1, ind2

def custom_select(pop, stagnation_counter, nd_select, k, forbidden_items=None):
    global STAGNATION_MODE
    STAGNATION_MODE = (stagnation_counter >= STAGNATION_LIMIT)
    target_size = k
    seen = set()
    num_tour = int(target_size * 0.70)  
    num_nsga2 = int(target_size * 0.25) 
    num_best = target_size - num_tour - num_nsga2 
    
    candidates = []
    candidates.extend(tools.selTournamentDCD(pop, num_tour))
    candidates.extend(tools.selNSGA2(pop, num_nsga2, nd=nd_select))
    candidates.extend(tools.selBest(pop, num_best))
    
    selected = []
    for ind in candidates:
        if tuple(ind) not in seen:
            seen.add(tuple(ind))
            selected.append(toolbox.clone(ind))
            if len(selected) == target_size: break
    if len(selected) < target_size:
        fillers = tools.selNSGA2(pop, target_size - len(selected))
        for f in fillers:
            child = toolbox.clone(f)
            if random.random() < 0.25:
                toolbox.mutate(child)
                del child.fitness.values
            selected.append(child)
    return selected[:target_size]

def analysis_file(pool, pop):
    global GLOBAL_MAX_SCORE
    toolbox.register("map", pool.map)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    update_crowding(pop)
    GLOBAL_MAX_SCORE = max([ind.fitness.values[0] for ind in pop])
    tqdm.write(f"\nInitial Max Count (Score): {GLOBAL_MAX_SCORE:.0f}", file=TQDM_FILE)
    return GLOBAL_MAX_SCORE

toolbox.register("mate", custom_mate)
toolbox.register("select", custom_select)

# =============================================================================
# 5. Mutation Strategies
# =============================================================================
EDGE_POSITIONS = None
def init_edge_positions():
    global EDGE_POSITIONS
    edges = []
    for c in range(IND_COLS): edges.append((0, c))
    for c in range(IND_COLS): edges.append((7, c))
    for r in range(1, IND_ROWS - 1): edges.append((r, 0))
    for r in range(1, IND_ROWS - 1): edges.append((r, IND_COLS - 1))
    EDGE_POSITIONS = np.array(edges, dtype=np.int64)

def directional_spread_mutation(grid):
    global GLOBAL_MAX_SCORE
    num_spread = 1 if GLOBAL_MAX_SCORE >= 3000 and not STAGNATION_MODE else rng.integers(1, 4)
    for _ in range(num_spread):
        if rng.random() < 0.5 and EDGE_POSITIONS is not None:
            idx = rng.integers(0, len(EDGE_POSITIONS))
            tr, tc = EDGE_POSITIONS[idx]
        else:
            tr = rng.integers(1, IND_ROWS - 1)
            tc = rng.integers(1, IND_COLS - 1)
        deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        dr, dc = deltas[rng.integers(0, 8)]
        sr, sc = tr + dr, tc + dc
        if 0 <= sr < IND_ROWS and 0 <= sc < IND_COLS:
            if rng.random() < 0.5:
                grid[tr, tc] = grid[sr, sc]
            else:
                grid[tr, tc] = rng.integers(0, 10)

def cyclic_remapping_mutation(grid):
    global GLOBAL_MAX_SCORE
    if GLOBAL_MAX_SCORE >= 3000 and not STAGNATION_MODE:
        k = rng.integers(1, 3)
    else:
        k = rng.integers(1, 4) if rng.random() < 0.7 else rng.integers(4, 6)
    selected = rng.choice(np.arange(10), size=k, replace=False)
    if k <= 2 or (k == 3 and rng.random() < 0.5):
        remaining = np.array([x for x in range(10) if x not in selected])
        perm = rng.choice(remaining, size=k, replace=False)
    else:
        shift = rng.integers(1, k)
        perm = np.roll(selected, shift)
    mapping = dict(zip(selected, perm))
    for i in range(IND_ROWS):
        for j in range(IND_COLS):
            val = grid[i, j]
            if val in mapping:
                grid[i, j] = mapping[val]

def adjacent_swap_mutation(grid):
    global GLOBAL_MAX_SCORE, EDGE_POSITIONS
    num_swaps = 1 if GLOBAL_MAX_SCORE >= 4000 else rng.integers(1, 4)
    deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for _ in range(num_swaps):
        if rng.random() < 0.5 and EDGE_POSITIONS is not None:
            idx = rng.integers(0, len(EDGE_POSITIONS))
            r, c = EDGE_POSITIONS[idx]
        else:
            r = rng.integers(0, IND_ROWS - 1)
            c = rng.integers(0, IND_COLS - 1)
        valid_neighbors = []
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < IND_ROWS and 0 <= nc < IND_COLS:
                valid_neighbors.append((nr, nc))
        if valid_neighbors:
            target_r, target_c = valid_neighbors[rng.integers(0, len(valid_neighbors))]
            grid[r, c], grid[target_r, target_c] = grid[target_r, target_c], grid[r, c]

def dlas_mutation(grid):
    grid_2d = np.array(grid).reshape(IND_ROWS, IND_COLS)
    with open("../data/grid.txt", "w") as f:
        for row in grid_2d:
            f.write(''.join(map(str, row)) + '\n')
    process = subprocess.Popen(
        [CPP_DLAS_BINARY],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None: break
        if line: tqdm.write(line.rstrip(), file=TQDM_FILE)
    process.wait()
    if os.path.exists("../data/result.txt"):
        with open("../data/result.txt", "r") as f:
            lines = [line.strip() for line in f if len(line.strip()) == 14]
            if len(lines) >= 8:
                new_flat = [int(d) for line in lines[:8] for d in line]
                grid[:] = np.array(new_flat).reshape(IND_ROWS, IND_COLS)
            else:
                tqdm.write("result.txt parsing failed", file=TQDM_FILE)
    else:
        tqdm.write("result.txt read failure.", file=TQDM_FILE)

def directional_cyclic_one_shift_mutation(grid):
    deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    r = rng.integers(0, IND_ROWS)
    c = rng.integers(0, IND_COLS)
    dr, dc = deltas[rng.integers(0, 8)]

    def get_line(r, c, dr, dc):
        line = []
        cr, cc = r - dr, c - dc
        while 0 <= cr < IND_ROWS and 0 <= cc < IND_COLS:
            line.insert(0, (cr, cc)); cr -= dr; cc -= dc
        line.append((r, c))
        cr, cc = r + dr, c + dc
        while 0 <= cr < IND_ROWS and 0 <= cc < IND_COLS:
            line.append((cr, cc)); cr += dr; cc += dc
        return line

    line = get_line(r, c, dr, dc)
    if len(line) < 3:
        r = rng.integers(1, IND_ROWS - 2)
        c = rng.integers(1, IND_COLS - 2)
        dr, dc = deltas[rng.integers(0, 8)]
        line = get_line(r, c, dr, dc)

    values = [grid[pr, pc] for pr, pc in line]
    shift = rng.integers(1, len(line))
    shifted = values[-shift:] + values[:-shift]
    for i, (pr, pc) in enumerate(line):
        grid[pr, pc] = shifted[i]

def local_3x3_rotate_mutation(grid):
    indpb_r = 0.01 * rng.integers(1, 4)
    for row in range(1, IND_ROWS - 1):
        for col in range(1, IND_COLS - 1):
            if rng.random() < indpb_r:
                deltas = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
                neighbors, positions = [], []
                for dr, dc in deltas:
                    nr, nc = row + dr, col + dc
                    neighbors.append(grid[nr, nc])
                    positions.append((nr, nc))
                if rng.random() < 0.5:
                    shifted = [neighbors[-1]] + neighbors[:-1]
                else:
                    shifted = neighbors[1:] + [neighbors[0]]
                for i, (nr, nc) in enumerate(positions):
                    grid[nr, nc] = shifted[i]

# =============================================================================
# 6. Mutation Config
# =============================================================================
MUTATION_TYPES = [
    directional_spread_mutation,
    cyclic_remapping_mutation,
    adjacent_swap_mutation
]
NORMAL_PROBS = [0.30, 0.40, 0.30]
STAGNATION_PROBS = [0.25, 0.50, 0.25]

def custom_mutate(individual, indpb=0.05):
    global STAGNATION_MODE
    grid = np.array(individual).reshape(IND_ROWS, IND_COLS)
    probs = STAGNATION_PROBS if STAGNATION_MODE else NORMAL_PROBS
    r, cum = random.random(), 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            MUTATION_TYPES[i](grid)
            break
    individual[:] = grid.ravel().tolist()
    return individual,

toolbox.register("mutate", custom_mutate, indpb=0.05)
toolbox.register("attr_int", lambda: int(rng.integers(INT_MIN, INT_MAX + 1)))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =============================================================================
# 7. Utilities & File I/O
# =============================================================================
def reset_random_seed():
    global rng
    rng = get_secure_rng()
    random.seed(int(rng.integers(0, 2 ** 63)))
    np.random.seed(int(rng.integers(0, 2 ** 32 - 1)))

def update_crowding(population):
    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    return [ind for front in fronts for ind in front]

def load_previous_best():
    loaded, protected = [], None
    if os.path.exists(FILENAME):
        with open(FILENAME, 'r') as f:
            lines = [l.strip() for l in f if len(l.strip()) == 14 and l.strip().isdigit()]
        for i in range(0, len(lines), 8):
            if i + 7 >= len(lines): break
            ind_list = [int(d) for row in lines[i:i + 8] for d in row]
            ind = creator.Individual(ind_list)
            if i == 0: protected = ind
            else: loaded.append(ind)
    return protected, loaded

def load_individuals_from_file():
    protected, loaded = load_previous_best()
    pop = ([protected] if protected else []) + loaded
    if len(pop) < TARGET_POP:
        pop.extend(toolbox.population(n=TARGET_POP - len(pop)))
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    update_crowding(pop)
    return tools.selBest(pop, TARGET_POP)

def save_result(pop, g):
    tqdm.write(f"\n\n{'=' * 60}\nEvolution finished after {g} generations.", file=TQDM_FILE)
    top_k = tools.selBest(pop, k=25)
    with open(FILENAME, 'a') as f:
        f.write(f"\n--- Final TOP {len(top_k)} ---\n")
        for rank, ind in enumerate(top_k, 500):
            tqdm.write(f"Rank {rank} - Score: {ind.fitness.values[0]:.0f}", file=TQDM_FILE)
            for row in np.array(ind).reshape(IND_ROWS, IND_COLS):
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
    tqdm.write(f"Final TOP saved to {FILENAME}", file=TQDM_FILE)

# =============================================================================
# 8. Mass Mutation (with per-individual timing)
# =============================================================================
def perform_mass_mutation(pop, mut_prob=0.4):
    global GLOBAL_MAX_SCORE
    tqdm.write(f"{'=' * 10} MASS MUTATION EVENT TRIGGERED {'=' * 10}", file=TQDM_FILE)

    best_idx = int(np.argmax([ind.fitness.values[0] if ind.fitness.valid else -1 for ind in pop]))
    best_ind = toolbox.clone(pop[best_idx])
    original_best_score = best_ind.fitness.values[0]
    tqdm.write(f"Protecting top individual at index {best_idx} (Score: {original_best_score:.0f})", file=TQDM_FILE)

    # Save best_ind before anything overwrites result.txt
    best_grid_before = np.array(best_ind).reshape(IND_ROWS, IND_COLS)
    with open("../data/best_protected.txt", "w") as f:
        for row in best_grid_before:
            f.write(''.join(map(str, row)) + '\n')

    mass_start = time.perf_counter()
    # Snapshot GLOBAL_MAX_SCORE before the loop so the final comparison
    # is not affected by eval_814_heuristic updating it mid-loop.
    global_max_before_loop = GLOBAL_MAX_SCORE

    for i, ind in enumerate(pop):
        score_before = ind.fitness.values[0] if ind.fitness.valid else float('nan')
        ind_snapshot = toolbox.clone(ind)  # snapshot before mutation

        ind_start = time.perf_counter()

        mutated = False
        if random.random() < mut_prob:
            toolbox.mutate(ind)
            mutated = True

        del ind.fitness.values

        grid = np.array(ind).reshape(IND_ROWS, IND_COLS)
        dlas_mutation(grid)
        ind[:] = grid.ravel().tolist()

        # Evaluate immediately so we can report the score delta
        new_fit = toolbox.evaluate(ind)
        ind.fitness.values = new_fit
        score_after = new_fit[0]

        elapsed = time.perf_counter() - ind_start
        delta = score_after - score_before
        delta_str = f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}"
        mut_tag = "(Mutated)" if mutated else "( Normal)"

        # If score decreased, revert to snapshot
        if score_after < score_before:
            pop[i] = ind_snapshot
            tqdm.write(
                f"  [{i+1:2d}/{len(pop)}] {mut_tag} "
                f"score: {score_before:6.0f} -> {score_after:6.0f} ({delta_str:>6})  "
                f"time: {elapsed:.2f}s  [REVERTED]",
                file=TQDM_FILE
            )
        else:
            tqdm.write(
                f"  [{i+1:2d}/{len(pop)}] {mut_tag} "
                f"score: {score_before:6.0f} -> {score_after:6.0f} ({delta_str:>6})  "
                f"time: {elapsed:.2f}s",
                file=TQDM_FILE
            )
            # Mid-loop best upgrade
            if score_after > original_best_score:
                tqdm.write(
                    f"  !! Mid-loop upgrade: {original_best_score:.0f} -> {score_after:.0f} "
                    f"(individual {i+1})",
                    file=TQDM_FILE
                )
                best_ind = toolbox.clone(ind)
                original_best_score = score_after
                best_idx = i

    # Re-evaluate any remaining invalids
    invalid = [ind for ind in pop if not ind.fitness.valid]
    if invalid:
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

    update_crowding(pop)
    new_max = max([ind.fitness.values[0] for ind in pop])
    total_elapsed = time.perf_counter() - mass_start

    if new_max > global_max_before_loop:
        tqdm.write(
            f"Mass Mutat... NEW RECORD?! >> {global_max_before_loop:.0f} -> {new_max:.0f}  "
            f"(total: {total_elapsed:.1f}s)",
            file=TQDM_FILE
        )
        GLOBAL_MAX_SCORE = new_max
        with open(FILENAME, 'a') as f:
            best_grid = np.array(
                toolbox.clone(pop[np.argmax([ind.fitness.values[0] for ind in pop])])
            ).reshape(IND_ROWS, IND_COLS)
            for row in best_grid:
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
    else:
        GLOBAL_MAX_SCORE = new_max
        tqdm.write(
            f"Mass Mutation complete >> New Max: {new_max:.0f}  "
            f"(total: {total_elapsed:.1f}s)\n{'=' * 51}",
            file=TQDM_FILE
        )

    return new_max

# =============================================================================
# 9. Generation & Main
# =============================================================================
def get_stagnation_limit(current_max):
    return 50000

def generation(g, pop, max_score_all_time, stagnation_counter, seed_counter, last_max):
    global GLOBAL_MAX_SCORE, STAGNATION_MODE, CURRENT_CXPB, CURRENT_MUTPB, NORMAL_PROBS

    if max_score_all_time >= 3000:
        CURRENT_CXPB = 0.85   
        CURRENT_MUTPB = 0.10  
        NORMAL_PROBS = [0.15, 0.70, 0.15] 
    else:
        CURRENT_CXPB = 0.5
        CURRENT_MUTPB = 0.35
        NORMAL_PROBS = [0.30, 0.40, 0.30]

    best_set = tools.selBest(pop, ELITE_BEST_SIZE)
    elites = list(map(toolbox.clone, best_set + tools.selNSGA2(
        [i for i in pop if i not in best_set], ELITE_SIZE)))
    forbidden = elites

    offspring_needed = TARGET_POP - len(elites)
    offspring_candidates = [p for p in tools.selBest(pop, len(pop))[len(elites):]]
    
    offspring = list(map(toolbox.clone, toolbox.select(
        offspring_candidates, stagnation_counter, 'standard',
        offspring_needed, forbidden_items=forbidden)))

    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CURRENT_CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values

    for mutant in offspring:
        if random.random() < CURRENT_MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    pop[:] = elites + offspring
    invalid = [ind for ind in pop if not ind.fitness.valid]
    if invalid:
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

    update_crowding(pop)
    current_max = max([ind.fitness.values[0] for ind in pop])

    if current_max > max_score_all_time:
        tqdm.write(f"!! NEW RECORD! Gen {g}: {max_score_all_time:.0f} -> {current_max:.0f}", file=TQDM_FILE)
        max_score_all_time = GLOBAL_MAX_SCORE = current_max
        with open(FILENAME, 'a') as f:
            best_grid = np.array(
                toolbox.clone(pop[np.argmax([i.fitness.values[0] for i in pop])])
            ).reshape(IND_ROWS, IND_COLS)
            for row in best_grid:
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
        stagnation_counter = 0
        seed_counter = 0
        last_max = current_max
    else:
        stagnation_counter += 1
        seed_counter += 1
        last_max = current_max

    current_limit = get_stagnation_limit(max_score_all_time)
    STAGNATION_MODE = (stagnation_counter >= current_limit)

    if seed_counter >= G_SEED_GROUP:
        reset_random_seed()
        tqdm.write(f"[Stagnation] Seed Reset Triggered.", file=TQDM_FILE)
        seed_counter = 0

    if STAGNATION_MODE:
        new_max = perform_mass_mutation(pop)
        stagnation_counter = 0
        last_max = new_max
        max_score_all_time = new_max

    return max_score_all_time, stagnation_counter, seed_counter, last_max

def main():
    init_edge_positions()
    reset_random_seed()
    pop = load_individuals_from_file()
    with multiprocessing.Pool() as pool:
        max_score_all_time = analysis_file(pool, pop)
        stagnation_counter = 0
        seed_counter = 0
        last_max = max_score_all_time

        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        pbar = tqdm(range(1, NGEN + 1), desc="Evolution", unit="gen",
                    bar_format=bar_format, ncols=157, file=TQDM_FILE)

        for g in pbar:
            max_score_all_time, stagnation_counter, seed_counter, last_max = generation(
                g, pop, max_score_all_time, stagnation_counter, seed_counter, last_max
            )
            if g % G_PRINT_GROUP == 0 or g == 1:
                all_scores = [ind.fitness.values[0] for ind in pop]
                avg = np.mean(all_scores)
                q1, med, q3 = np.percentile(all_scores, [25, 50, 75])
                tqdm.write(
                    f"[Gen {g:7d}] Max : {max_score_all_time:6.0f} | Avg : {avg:6.1f} | "
                    f"Q1/Med/Q3 : {q1:6.1f} / {med:6.1f} / {q3:6.1f} | Stg : {stagnation_counter:7d}",
                    file=TQDM_FILE
                )
            pbar.set_description(f"[Max {max_score_all_time:.0f}]")

        save_result(pop, NGEN)

if __name__ == "__main__":
    main()