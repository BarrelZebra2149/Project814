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
FILENAME = 'final_gen_sep1.txt'

TARGET_POP = 100
NGEN = 500000
G_PRINT_GROUP = 10
G_STAG_GROUP = 200
STAGNATION_LIMIT = 100

ELITE_SIZE = 5
ELITE_BEST_SIZE = 10

CURRENT_CXPB = 0.3
CURRENT_MUTPB = 0.9

GLOBAL_MAX_SCORE = 0.0
STAGNATION_MODE = False

# =============================================================================
# 2. DEAP Framework Setup
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
    """2D block crossover."""
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)
    sy, sx = random.randint(0, IND_ROWS - 2), random.randint(0, IND_COLS - 2)
    ey, ex = random.randint(sy + 1, IND_ROWS), random.randint(sx + 1, IND_COLS)
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
    if r < 0.40:
        candidates = tools.selBest(pop, select_size)
    elif r < 0.50:
        candidates = tools.selRandom(pop, select_size)
    elif r < 0.65:
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
    sys.stdout.write(f"Initial Max Count (Score): {GLOBAL_MAX_SCORE:.0f}\n")
    sys.stdout.flush()
    return GLOBAL_MAX_SCORE


toolbox.register("mate", custom_mate)
toolbox.register("select", custom_select)


# =============================================================================
# 5. Mutation Strategies
# =============================================================================
def directional_spread_mutation(grid):
    """Spreads existing digits to neighboring cells."""
    indpb_r = 0.01 * random.randint(1, 3)
    for row in range(IND_ROWS):
        for col in range(IND_COLS):
            if random.random() < indpb_r:
                deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                possible = [(dr, dc) for dr, dc in deltas if 0 <= row + dr < IND_ROWS and 0 <= col + dc < IND_COLS]
                if possible:
                    dr, dc = random.choice(possible)
                    if (dr * dc != 0) != (random.random() < 0.5):
                        grid[row + dr, col + dc] = grid[row, col]
                    else:
                        grid[row + dr, col + dc] = random.randint(0, 9)


def cyclic_remapping_mutation(grid):
    """Shuffles a selected subset of digits globally."""
    k = random.randint(1, 10)
    selected = random.sample(range(10), k)
    if k <= 2 or (3 <= k < 5 and random.random() < 0.5):
        remaining = [x for x in range(10) if x not in selected]
        perm = random.sample(remaining, k)
    else:
        shift = random.randint(1, k - 1)
        perm = selected[-shift:] + selected[:-shift]
    mapping = dict(zip(selected, perm))
    for i in range(IND_ROWS):
        for j in range(IND_COLS):
            if grid[i, j] in mapping:
                grid[i, j] = mapping[grid[i, j]]


def directional_cyclic_one_shift_mutation(grid):
    """Cyclically shifts a line by random amount."""
    rows, cols = grid.shape
    mutated = False
    while not mutated:
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        dr, dc = random.choice(deltas)
        line_coords = []
        cr, cc = r, c
        while 0 <= cr < rows and 0 <= cc < cols:
            line_coords.append((cr, cc))
            cr += dr
            cc += dc
        cr, cc = r - dr, c - dc
        while 0 <= cr < rows and 0 <= cc < cols:
            line_coords.insert(0, (cr, cc))
            cr -= dr
            cc -= dc
        if len(line_coords) >= 3:
            values = [grid[pr, pc] for pr, pc in line_coords]
            shift = random.randint(1, len(line_coords) - 1)
            shifted_values = values[-shift:] + values[:-shift]
            for i, (pr, pc) in enumerate(line_coords):
                grid[pr, pc] = shifted_values[i]
            mutated = True


def quadrant_border_rotate_mutation(grid):
    """Rotates the border of one quadrant."""
    rows, cols = grid.shape
    mutated = False
    while not mutated:
        rt = random.randint(1, rows - 2)
        ct = random.randint(1, cols - 2)
        quads = [(0, rt, 0, ct), (0, rt, ct + 1, cols - 1),
                 (rt + 1, rows - 1, 0, ct), (rt + 1, rows - 1, ct + 1, cols - 1)]
        r1, r2, c1, c2 = random.choice(quads)
        border = []
        for c in range(c1, c2 + 1): border.append((r1, c))
        for r in range(r1 + 1, r2 + 1): border.append((r, c2))
        for c in range(c2 - 1, c1 - 1, -1): border.append((r2, c))
        for r in range(r2 - 1, r1, -1): border.append((r, c1))
        if len(border) >= 4:
            values = [grid[pr, pc] for pr, pc in border]
            if random.random() < 0.5:
                shifted = [values[-1]] + values[:-1]
            else:
                shifted = values[1:] + [values[0]]
            for i, (pr, pc) in enumerate(border):
                grid[pr, pc] = shifted[i]
            mutated = True


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
    directional_cyclic_one_shift_mutation,
    quadrant_border_rotate_mutation,
    full_digit_cycle_mutation
]

NORMAL_PROBS = [0.30, 0.30, 0.05, 0.05, 0.30]
STAGNATION_PROBS = [0.20, 0.20, 0.20, 0.20, 0.20]


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
    """Forces hardware-level randomness."""
    seed_bytes = os.urandom(4)
    new_seed = int.from_bytes(seed_bytes, byteorder='big')
    random.seed(new_seed)
    np.random.seed(new_seed)


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
    return tools.selBest(pop, TARGET_POP)


def save_result(pop, g):
    """Saves final top individuals to file."""
    sys.stdout.write(f"\n{'=' * 60}\nEvolution finished after {g} generations.\n")
    top_k = tools.selBest(pop, k=ELITE_SIZE * 4)
    with open(FILENAME, 'a') as f:
        f.write(f"\n--- Final TOP {len(top_k)} ---\n")
        for rank, ind in enumerate(top_k, 1):
            sys.stdout.write(f"Rank {rank} - Score: {ind.fitness.values[0]:.0f}\n")
            for row in np.array(ind).reshape(IND_ROWS, IND_COLS):
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
    sys.stdout.write(f"Final TOP saved to {FILENAME}\n")


# =============================================================================
# 8. DLAS (Score Focus)
# =============================================================================
@njit(fastmath=True)
def _full_digit_cycle_numba(grid):
    """Numba version of full_digit_cycle_mutation."""
    digits = np.arange(10)
    np.random.shuffle(digits)
    mapping = np.empty(10, dtype=np.int64)
    for i in range(10):
        mapping[digits[i]] = digits[(i + 1) % 10]
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = mapping[grid[i, j]]


@njit(fastmath=True)
def _dlas_score_core_numba(initial_grid, initial_score, max_idle, rows, cols):
    """Numba-accelerated DLAS core for consecutive score."""
    states = np.empty((3, rows, cols), dtype=np.int64)
    states[0] = initial_grid.copy()
    states[1] = initial_grid.copy()
    states[2] = initial_grid.copy()

    scores = np.array([initial_score, initial_score, initial_score], dtype=np.float64)
    cur_idx = 0
    best_idx = 0
    L = 20
    history = np.full(L, initial_score, dtype=np.float64)
    h_idx = 0
    idle_iter = 0

    while idle_iter < max_idle:
        cand_idx = (cur_idx + 1) % 3
        if cand_idx == best_idx:
            cand_idx = (cand_idx + 1) % 3
        states[cand_idx][:] = states[cur_idx]

        if np.random.rand() < 0.9:
            r, c = np.random.randint(0, rows), np.random.randint(0, cols)
            states[cand_idx, r, c] = np.random.randint(0, 10)
        else:
            for _ in range(3):
                r, c = np.random.randint(0, rows), np.random.randint(0, cols)
                states[cand_idx, r, c] = np.random.randint(0, 10)

        new_score, _ = _evaluate_core_numba(states[cand_idx].ravel(), rows, cols)

        if new_score > scores[best_idx]:
            scores[cand_idx] = new_score
            best_idx = cur_idx = cand_idx
            idle_iter = 0
        elif new_score >= history[h_idx] or new_score >= scores[cur_idx]:
            scores[cand_idx] = new_score
            cur_idx = cand_idx
            idle_iter += 1
        else:
            idle_iter += 1

        history[h_idx] = scores[cur_idx]
        h_idx = (h_idx + 1) % L

        if idle_iter == int(max_idle * 0.4):
            cur_idx = best_idx
            _full_digit_cycle_numba(states[cur_idx])
            new_score, _ = _evaluate_core_numba(states[cur_idx].ravel(), rows, cols)
            scores[cur_idx] = new_score

    return states[best_idx].ravel(), scores[best_idx]


def perform_dlas_score_local_search(pop):
    """DLAS optimized for consecutive score."""
    global GLOBAL_MAX_SCORE
    best_ind = tools.selBest(pop, 1)[0]
    initial_grid = np.array(best_ind).reshape(IND_ROWS, IND_COLS).copy()
    initial_score = float(best_ind.fitness.values[0])

    sys.stdout.write(f"\nDLAS (Score Focus) Started | Initial Score: {initial_score:.0f}\n")
    sys.stdout.flush()

    best_grid_1d, best_score = _dlas_score_core_numba(initial_grid, initial_score, 100, IND_ROWS, IND_COLS)

    best_ind[:] = best_grid_1d.tolist()
    del best_ind.fitness.values
    best_ind.fitness.values = toolbox.evaluate(best_ind)
    update_crowding(pop)

    current_max_score = max([ind.fitness.values[0] for ind in pop])
    sys.stdout.write(f"DLAS Finished | Best Score: {current_max_score:.0f}\n")
    sys.stdout.flush()

    GLOBAL_MAX_SCORE = current_max_score
    return GLOBAL_MAX_SCORE


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
    current_max = max([ind.fitness.values[0] for ind in pop])

    # Record check
    if current_max > max_score_all_time:
        sys.stdout.write(f"!! NEW RECORD! Gen {g}: {max_score_all_time:.0f} -> {current_max:.0f}\n")
        sys.stdout.flush()
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

    # Seed reset
    if seed_counter > G_STAG_GROUP:
        reset_random_seed()
        seed_counter = 0

    # Logging
    if g % G_PRINT_GROUP == 0:
        all_scores = [ind.fitness.values[0] for ind in pop]
        current_avg = np.mean(all_scores)
        q1, median, q3 = np.percentile(all_scores, [25, 50, 75])
        sys.stdout.write(
            f"Gen {g:5d} | Max: {current_max:>6.0f} | Avg: {current_avg:>6.1f} | "
            f"Q3: {q3:>6.1f} | Med: {median:>6.1f} | Q1: {q1:>6.1f}\n"
        )
        sys.stdout.flush()

    # DLAS trigger
    if stagnation_counter >= STAGNATION_LIMIT:
        new_max = perform_dlas_score_local_search(pop)
        stagnation_counter = 0
        last_max = new_max
        max_score_all_time = new_max

    return max_score_all_time, stagnation_counter, seed_counter, last_max


def main():
    pop = load_individuals_from_file()
    with multiprocessing.Pool() as pool:
        max_score_all_time = analysis_file(pool, pop)
        stagnation_counter = 0
        seed_counter = 0
        last_max = max_score_all_time
        for g in range(1, NGEN + 1):
            max_score_all_time, stagnation_counter, seed_counter, last_max = generation(
                g, pop, max_score_all_time, stagnation_counter, seed_counter, last_max
            )
        save_result(pop, NGEN)


if __name__ == "__main__":
    main()