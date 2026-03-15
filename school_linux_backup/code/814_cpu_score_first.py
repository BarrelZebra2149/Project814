import random
import numpy as np
import sys
import os
import multiprocessing
import subprocess
import time
from deap import base, creator, tools
from numba import njit
# =============================================================================
# 1. Hyperparameters & Global Constants
# =============================================================================
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = '../data/test_gen6.txt'
TARGET_POP = 25
NGEN = 5000000
G_PRINT_GROUP = 1000
G_SEED_GROUP = 200000
STAGNATION_LIMIT = 1000000
ELITE_SIZE = 5
ELITE_BEST_SIZE = 5
CURRENT_CXPB = 0.5
CURRENT_MUTPB = 0.8
GLOBAL_MAX_SCORE = 0.0
STAGNATION_MODE = False
CPP_DLAS_BINARY = "./my_dlas"
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
    """Ultra-fast pathfinding using static stack."""
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
    """Extract digits using math only."""
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
    """Reverse integer using math only."""
    rev = 0
    while n > 0:
        rev = rev * 10 + (n % 10)
        n //= 10
    return rev
@njit
def _evaluate_core_numba(grid_1d, rows, cols):
    """Returns (current_score, formable)."""
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
    return float(current_score), float(formable)
def eval_814_heuristic(individual):
    """Fitness: (current_score, formable) - Score is prioritized."""
    global GLOBAL_MAX_SCORE
    grid_1d = np.array(individual, dtype=np.int64)
    current_score, formable = _evaluate_core_numba(grid_1d, IND_ROWS, IND_COLS)
    if current_score > GLOBAL_MAX_SCORE:
        GLOBAL_MAX_SCORE = current_score
    return float(current_score), float(formable)
toolbox.register("evaluate", eval_814_heuristic)
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
    """Custom selection with elite protection."""
    global STAGNATION_MODE, CURRENT_CXPB, CURRENT_MUTPB
    STAGNATION_MODE = (stagnation_counter >= STAGNATION_LIMIT)
    target_size = k
    seen = set()
    if forbidden_items and not STAGNATION_MODE:
        for f_ind in forbidden_items:
            seen.add(tuple(f_ind))
    select_size = int(len(pop) * 0.9)
    r = random.random()
    if r < 0.45:
        candidates = tools.selBest(pop, select_size)
    elif r < 0.50:
        candidates = tools.selWorst(pop, select_size)
    elif r < 0.55:
        candidates = tools.selNSGA2(pop, select_size, nd=nd_select)
    else:
        candidates = tools.selTournamentDCD(pop, select_size)
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
    """Evaluates the initial population using multiprocessing."""
    global GLOBAL_MAX_SCORE
    toolbox.register("map", pool.map)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    update_crowding(pop)
    # formable (count) is values[0]
    GLOBAL_MAX_SCORE = max([ind.fitness.values[0] for ind in pop])
    sys.stdout.write(f"\nInitial Max Count (Score): {GLOBAL_MAX_SCORE:.0f}")
    sys.stdout.flush()
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
    for c in range(IND_COLS):
        edges.append((0, c))
    for c in range(IND_COLS):
        edges.append((7, c))
    for r in range(1, IND_ROWS - 1):
        edges.append((r, 0))
    for r in range(1, IND_ROWS - 1):
        edges.append((r, IND_COLS - 1))
    EDGE_POSITIONS = np.array(edges, dtype=np.int64)
    
def directional_spread_mutation(grid):
    if rng.random() < 0.5:
        num_spread = 1
    else:
        num_spread = rng.integers(2, 4)
        
    for _ in range(num_spread):
        if rng.random() < 0.5 and EDGE_POSITIONS is not None:
            idx = rng.integers(0, len(EDGE_POSITIONS))
            tr, tc = EDGE_POSITIONS[idx]
        else:
            tr = rng.integers(1, IND_ROWS - 1)
            tc = rng.integers(1, IND_COLS - 1)
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        dr, dc = deltas[rng.integers(0, 8)]
        sr, sc = tr + dr, tc + dc
        if 0 <= sr < IND_ROWS and 0 <= sc < IND_COLS:
            if rng.random() < 0.5:
                grid[tr, tc] = grid[sr, sc]
            else:
                grid[tr, tc] = rng.integers(0, 10)
                
def cyclic_remapping_mutation(grid):
    if rng.random() < 0.5:
        k = rng.integers(1, 3)
    else:
        k = rng.integers(3, 8)
    
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

# ====================== NEW: C++ DLAS Mutation ======================
def dlas_mutation(grid):
    """
    C++ DLAS? ???? mutation
    - grid.txt? ?? ?? ??
    - ./my_dlas ?? (1? DLAS)
    - result.txt?? ??? ?? ???
    """
    grid_2d = np.array(grid).reshape(IND_ROWS, IND_COLS)
    with open("../data/grid.txt", "w") as f:
        for row in grid_2d:
            f.write(''.join(map(str, row)) + '\n')

    with open("dlas_log.txt", "w") as log_file:
        print("\n=== dlas started ===")
        process = subprocess.Popen(
            [CPP_DLAS_BINARY],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # ??? stdout?? ??
            text=True,
            bufsize=1
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())         # ???? ??? ??
                log_file.write(line)        # ???? ??
                log_file.flush()            # ?? ??

        process.wait()
        print("=== C++ DLAS ?? ===\n")
        print("?? ??: dlas_log.txt")
    if os.path.exists("../data/result.txt"):
        with open("../data/result.txt", "r") as f:
            lines = [line.strip() for line in f if len(line.strip()) == 14]
            if len(lines) >= 8:
                new_flat = [int(d) for line in lines[:8] for d in line]
                grid[:] = np.array(new_flat).reshape(IND_ROWS, IND_COLS)   
                print("DLAS mutation finished")
            else:
                print("result.txt parsing failed")
    else:
        print("result.txt read failure.")

# =============================================================================
# 6. Mutation Config - DLAS 50% + ??? 25%?
# =============================================================================
MUTATION_TYPES = [
    directional_spread_mutation,
    cyclic_remapping_mutation,
    dlas_mutation
]

NORMAL_PROBS = [0.25, 0.25, 0.50]      # directional 25%, cyclic 25%, DLAS 50%
STAGNATION_PROBS = [0.25, 0.25, 0.50]
def custom_mutate(individual, indpb=0.05):
    """Main mutation dispatcher using roulette wheel."""
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
    """Assigns crowding distance for NSGA-II."""
    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    return [ind for front in fronts for ind in front]
def load_previous_best():
    """Loads saved best grids from file."""
    loaded, protected = [], None
    if os.path.exists(FILENAME):
        with open(FILENAME, 'r') as f:
            lines = [l.strip() for l in f if len(l.strip()) == 14 and l.strip().isdigit()]
        for i in range(0, len(lines), 8):
            if i + 7 >= len(lines): break
            ind_list = [int(d) for row in lines[i:i + 8] for d in row]
            ind = creator.Individual(ind_list)
            if i == 0:
                protected = ind
            else:
                loaded.append(ind)
    return protected, loaded
def load_individuals_from_file():
    """Builds initial population from saved grids + random ones."""
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
    """Saves final top individuals to file."""
    tqdm.write(f"\n\n{'=' * 60}\nEvolution finished after {g} generations.")
    top_k = tools.selBest(pop, k=50)
    with open(FILENAME, 'a') as f:
        f.write(f"\n--- Final TOP {len(top_k)} ---\n")
        for rank, ind in enumerate(top_k, 1):
            tqdm.write(f"Rank {rank} - Score: {ind.fitness.values[0]:.0f}")
            for row in np.array(ind).reshape(IND_ROWS, IND_COLS):
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
    tqdm.write(f"Final TOP saved to {FILENAME}")
def perform_mass_mutation(pop):
    """
    Mass Mutation Event triggered strictly during extreme stagnation.
    Removes elite protection, forcing one round of mutation on the entire population,
    and immediately re-evaluates all individuals.
    @param pop: The current population list.
    @return: The newly found maximum score after the mass mutation event.
    """
    global GLOBAL_MAX_SCORE
    tqdm.write(f"{'=' * 10} MASS MUTATION EVENT TRIGGERED {'=' * 10}")
    tqdm.write(f"Elite protection disabled. Forcing mutation on all {len(pop)} individuals.")
    sys.stdout.flush()
    for ind in pop:
        toolbox.mutate(ind)
        del ind.fitness.values
    invalid = [ind for ind in pop if not ind.fitness.valid]
    if invalid:
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
    update_crowding(pop)
    new_max = max([ind.fitness.values[0] for ind in pop])
    GLOBAL_MAX_SCORE = new_max
    tqdm.write(f"Mass Mutation complete >> New Max: {new_max:.0f}\n{'=' * 51}")
    sys.stdout.flush()
    return new_max
# =============================================================================
# 9. Generation & Main
# =============================================================================

def get_stagnation_limit(current_max):
    limit = (10 * current_max) + ((current_max / 300) ** 4.5)
    return int(limit)

def generation(g, pop, max_score_all_time, stagnation_counter, seed_counter, last_max):
    """Single generation step."""
    global GLOBAL_MAX_SCORE, STAGNATION_MODE
    
    # Elite Preservation
    best_set = tools.selBest(pop, ELITE_BEST_SIZE)
    elites = list(map(toolbox.clone, best_set + tools.selNSGA2(
        [i for i in pop if i not in best_set], ELITE_SIZE)))
    forbidden = elites
    
    # Offspring
    offspring_candidates = [p for p in tools.selBest(pop, len(pop))[ELITE_SIZE:]]
    offspring = list(map(toolbox.clone, toolbox.select(
        offspring_candidates, stagnation_counter, 'standard',
        TARGET_POP - ELITE_SIZE, forbidden_items=forbidden)))
        
    # Crossover
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CURRENT_CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values
            
    # Mutation
    for mutant in offspring:
        if random.random() < CURRENT_MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
    # Evaluation
    invalid = [ind for ind in pop + offspring if not ind.fitness.valid]
    if invalid:
        for ind, fit in zip(invalid, list(toolbox.map(toolbox.evaluate, invalid))):
            ind.fitness.values = fit
    pop[:] = elites + offspring
    update_crowding(pop)
    current_max = max([ind.fitness.values[0] for ind in pop])
    
    # Record check
    if current_max > max_score_all_time:
        tqdm.write(f"!! NEW RECORD! Gen {g}: {max_score_all_time:.0f} -> {current_max:.0f}")
        max_score_all_time = GLOBAL_MAX_SCORE = current_max
        with open(FILENAME, 'a') as f:
            best_grid = np.array(toolbox.clone(pop[np.argmax([i.fitness.values[0] for i in pop])])).reshape(IND_ROWS,
                                                                                                            IND_COLS)
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
        pbar.write(f"[Stagnation] Seed Reset Triggered.")
        seed_counter = 0

    if stagnation_counter >= current_limit:
        new_max = perform_mass_mutation(pop)
        stagnation_counter = 0
        last_max = new_max
        max_score_all_time = new_max
    return max_score_all_time, stagnation_counter, seed_counter, last_max

from tqdm import tqdm
import numpy as np
import shutil
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
        pbar = tqdm(range(1, NGEN + 1), desc="Evolution", unit="gen", bar_format=bar_format, ncols=157)  
        for g in pbar:
            max_score_all_time, stagnation_counter, seed_counter, last_max = generation(
                g, pop, max_score_all_time, stagnation_counter, seed_counter, last_max
            )
            if g % G_PRINT_GROUP == 0 or g == 1:
                all_scores = [ind.fitness.values[0] for ind in pop]
                avg = np.mean(all_scores)
                q1, med, q3 = np.percentile(all_scores, [25, 50, 75])
                
                pbar.write(f"[Gen {g:7d}] Max : {max_score_all_time:6.0f} | Avg : {avg:6.1f} | "
                           f"Q1/Med/Q3 : {q1:6.1f} / {med:6.1f} / {q3:6.1f} | Stg : {stagnation_counter:7d}")
            pbar.set_description(f"[Max {max_score_all_time:.0f}]")
        save_result(pop, NGEN)
if __name__ == "__main__":
    main()