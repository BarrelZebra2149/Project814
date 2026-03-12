import random
import numpy as np
import sys
import os
import multiprocessing
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
FILENAME = 'new_gen.txt'

TARGET_POP = 70
NGEN = 2000000
G_PRINT_GROUP = 1000
STAGNATION_LIMIT = 30000
ELITE_SIZE = 5
ELITE_BEST_SIZE = 10
CURRENT_CXPB = 0.3
CURRENT_MUTPB = 0.8
GLOBAL_MAX_SCORE = 0.0
STAGNATION_MODE = False
# =============================================================================
# 2. DEAP Setup
# =============================================================================
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# =============================================================================
# 3. Core Evaluation (Numba Optimized)
# =============================================================================
@njit(fastmath=True)
def _has_path_fast(grid, digits):
    """Ultra-fast pathfinding using static stack."""
    rows, cols = grid.shape
    digit_len = digits.shape[0]
    deltas = np.array([[-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]], dtype=np.int64)
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
    return float(formable), float(current_score)


def eval_814_heuristic(individual):
    """Fitness: (formable, current_score)"""
    global GLOBAL_MAX_SCORE
    grid_1d = np.array(individual, dtype=np.int64)
    current_score, formable = _evaluate_core_numba(grid_1d, IND_ROWS, IND_COLS)
    if current_score > GLOBAL_MAX_SCORE:
        GLOBAL_MAX_SCORE = current_score
    return float(formable), float(current_score)


toolbox.register("evaluate", eval_814_heuristic)

# =============================================================================
# 4. Custom Genetic Operators
# =============================================================================
def custom_mate(ind1, ind2):
    """2D block crossover."""
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)
    sy, sx = random.randint(0, IND_ROWS-2), random.randint(0, IND_COLS-2)
    ey, ex = random.randint(sy+1, IND_ROWS), random.randint(sx+1, IND_COLS)
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
    
    select_size = int(len(pop) * 0.7)
    r = random.random()
    if r < 0.40: candidates = tools.selBest(pop, select_size)
    elif r < 0.50: candidates = tools.selRandom(pop, select_size)
    elif r < 0.65: candidates = tools.selNSGA2(pop, select_size, nd=nd_select)
    else: candidates = tools.selTournamentDCD(pop, select_size)
    
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

    for r in range(1, IND_ROWS-1):
        edges.append((r, 0))

    for r in range(1, IND_ROWS-1):
        edges.append((r, IND_COLS-1))
    
    EDGE_POSITIONS = np.array(edges, dtype=np.int64)


def directional_spread_mutation(grid):
    num_spread = random.randint(1, 5)
    
    for _ in range(num_spread):
        if random.random() < 0.5 and EDGE_POSITIONS is not None:
            idx = random.randint(0, len(EDGE_POSITIONS) - 1)
            tr, tc = EDGE_POSITIONS[idx]
        else:
            tr = random.randint(1, IND_ROWS - 2)
            tc = random.randint(1, IND_COLS - 2)
        
        deltas = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        possible = [(tr + dr, tc + dc) for dr, dc in deltas 
                    if 0 <= tr + dr < IND_ROWS and 0 <= tc + dc < IND_COLS]
        
        if not possible:
            continue
            
        sr, sc = random.choice(possible)
        
        if random.random() < 0.5:
            grid[tr, tc] = grid[sr, sc]
        else:
            grid[tr, tc] = random.randint(0, 9)


def cyclic_remapping_mutation(grid):
    """Shuffles a selected subset of digits globally."""
    k = random.randint(1, 7)
    selected = random.sample(range(10), k)
    if k <= 2 or (3 <= k < 5 and random.random() < 0.5):
        remaining = [x for x in range(10) if x not in selected]
        perm = random.sample(remaining, k)
    else:
        shift = random.randint(1, k-1)
        perm = selected[-shift:] + selected[:-shift]
    mapping = dict(zip(selected, perm))
    for i in range(IND_ROWS):
        for j in range(IND_COLS):
            if grid[i, j] in mapping:
                grid[i, j] = mapping[grid[i, j]]


def full_digit_cycle_mutation(grid):
    """Strong global remapping: cyclic shift of all 10 digits."""
    digits = list(range(10))
    random.shuffle(digits)
    mapping = {digits[i]: digits[(i + 1) % 10] for i in range(10)}
    for i in range(IND_ROWS):
        for j in range(IND_COLS):
            grid[i, j] = mapping[grid[i, j]]


# =============================================================================
# 6. Mutation Config
# =============================================================================
MUTATION_TYPES = [
    directional_spread_mutation,
    cyclic_remapping_mutation,
    full_digit_cycle_mutation
]

NORMAL_PROBS = [0.30, 0.40, 0.30]
STAGNATION_PROBS = [1.00, 0.00, 0.00]



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
toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =============================================================================
# 7. Utilities & File I/O
# =============================================================================
def reset_random_seed():
    seed_bytes = os.urandom(4)
    hw_seed = int.from_bytes(seed_bytes, byteorder='big')
    time_seed = int(time.time() * 1000)
    combined_seed = hw_seed ^ time_seed
    
    random.seed(combined_seed)
    np.random.seed(combined_seed % (2**32))


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
            ind_list = [int(d) for row in lines[i:i+8] for d in row]
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

# =============================================================================
# 8. DLAS (Formable Count Focus)
# =============================================================================

def analysis_file(pool, pop):
    """Evaluates the initial population using multiprocessing."""
    global GLOBAL_MAX_SCORE
    toolbox.register("map", pool.map)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    update_crowding(pop)
    
    # formable (count) is values[1]
    GLOBAL_MAX_SCORE = max([ind.fitness.values[1] for ind in pop])
    sys.stdout.write(f"Initial Max Count (Formable): {GLOBAL_MAX_SCORE:.0f}\n")
    sys.stdout.flush()
    return GLOBAL_MAX_SCORE
        

def perform_mass_mutation(pop):
    """
    Mass Mutation Event triggered strictly during extreme stagnation.
    Removes elite protection, forcing one round of mutation on the entire population,
    and immediately re-evaluates all individuals.

    @param pop: The current population list.
    @return: The newly found maximum score after the mass mutation event.
    """
    global GLOBAL_MAX_SCORE
    pbar.write(f"\n{'=' * 10} MASS MUTATION EVENT TRIGGERED {'=' * 10}\n")
    pbar.write(f"Elite protection disabled. Forcing mutation on all {len(pop)} individuals.\n")
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

    pbar.write(f"Mass Mutation complete >> New Max: {new_max:.0f}\n{'=' * 51}\n")
    sys.stdout.flush()
    return new_max

# =============================================================================
# 9. Generation & Main
# =============================================================================
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
    current_max = max([ind.fitness.values[1] for ind in pop])
    
    # Record check
    if current_max > max_score_all_time:
        tqdm.write(f"!! NEW RECORD! Gen {g}: {max_score_all_time:.0f} -> {current_max:.0f}")
        max_score_all_time = GLOBAL_MAX_SCORE = current_max
        with open(FILENAME, 'a') as f:
            best_grid = np.array(toolbox.clone(pop[np.argmax([i.fitness.values[1] for i in pop])])).reshape(IND_ROWS, IND_COLS)
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
    
    # DLAS trigger
    if stagnation_counter >= STAGNATION_LIMIT:
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
                all_scores = [ind.fitness.values[1] for ind in pop]
                avg = np.mean(all_scores)
                q1, med, q3 = np.percentile(all_scores, [25, 50, 75])
                
                pbar.write(f"[Gen {g:7d}] Max : {max_score_all_time:6.0f} | Avg : {avg:6.1f} | "
                           f"Q1/Med/Q3 : {q1:6.1f} / {med:6.1f} / {q3:6.1f} | Stag:{stagnation_counter:6d}")
            pbar.set_description(f"[Max {max_score_all_time:.0f}]")
        save_result(pop, NGEN)

if __name__ == "__main__":
    main()