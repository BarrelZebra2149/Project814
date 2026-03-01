import random
import numpy as np
import sys
import os
import multiprocessing
import time
from deap import base, creator, tools
from numba import njit
import binascii

# =============================================================================
# 1. Hyperparameters & Global Constants
# =============================================================================
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = 'new_gen0.txt'
TARGET_POP = 80
NGEN = 500000

G_PRINT_GROUP = 1000  # Console log interval
G_STAG_GROUP = 150  # Interval to reset random seed on stagnation
STAGNATION_LIMIT = 10000  # Limit to trigger Mass Mutation
ELITE_SIZE = 10
ELITE_BEST_SIZE = 7

# Dynamic probabilities (Updated during evolution)
CURRENT_CXPB = 0.5
CURRENT_MUTPB = 0.6
MASS_MUTATE_PENDING = False
STAGNATION_MODE = False  # Tracks if the population is currently stagnating
GLOBAL_MAX_SCORE = 0.0

# =============================================================================
# 2. DEAP Framework Setup
# =============================================================================
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


# =============================================================================
# 3. Core Pathfinding & Evaluation (Numba Optimized)
# =============================================================================
@njit
def _has_path(grid, digits):
    """
    Fast pathfinding using Numba. Checks if a specific sequence of digits
    can be traced in the grid using 8-way movement.

    @param grid: 2D numpy array representing the current individual.
    @param digits: 1D numpy array of digits to find in sequence.
    @return: True if the path exists, False otherwise.
    """
    rows, cols = grid.shape
    digit_len = digits.shape[0]
    if digit_len == 0:
        return False

    deltas = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    stack = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == digits[0]:
                stack.append((r, c, 0))

    while stack:
        r, c, idx = stack.pop()
        if idx == digit_len - 1:
            return True

        next_digit = digits[idx + 1]
        for i in range(8):
            nr, nc = r + deltas[i, 0], c + deltas[i, 1]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == next_digit:
                stack.append((nr, nc, idx + 1))
    return False


def count_occurrences(grid, n):
    """
    Wrapper for the Numba pathfinder to convert an integer into a digit array.

    @param grid: 2D numpy array.
    @param n: Integer number to search for in the grid.
    @return: 1 if the number can be formed, 0 otherwise.
    """
    digits = np.array([int(d) for d in str(n)], dtype=np.int64)
    return 1 if _has_path(grid, digits) else 0


def eval_814_heuristic(individual):
    """
    Evaluates the grid based on consecutive numbers found starting from 1.

    @param individual: A 1D list representing the flattened grid.
    @return: A tuple of floats (Current Score, Formable Potential).
    """
    global GLOBAL_MAX_SCORE
    grid = np.array(individual, dtype=np.int64).reshape(IND_ROWS, IND_COLS)
    found_set, n = set(), 1

    while n < 50000:
        rev_n = int(str(n)[::-1])
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
        current_score = 49999

    if current_score > GLOBAL_MAX_SCORE:
        GLOBAL_MAX_SCORE = current_score

    formable = max(0, min(9999, current_score) - 1000 + 1)
    for num in range(max(1000, current_score + 1), 10000):
        rev_num = int(str(num)[::-1])
        if num in found_set or (num % 10 != 0 and rev_num in found_set):
            formable += 1
            continue
        if count_occurrences(grid, num):
            formable += 1
            found_set.add(num)
            found_set.add(rev_num)

    return float(current_score), float(formable)


toolbox.register("evaluate", eval_814_heuristic)


# =============================================================================
# 4. Custom Genetic Operators (Mate, Select)
# =============================================================================
def custom_mate(ind1, ind2):
    """
    Performs a 2D block-based crossover between two parent grids.

    @param ind1: First parent individual.
    @param ind2: Second parent individual.
    @return: A tuple containing the two modified offspring (ind1, ind2).
    """
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)

    sy, sx = random.randint(0, IND_ROWS - 2), random.randint(0, IND_COLS - 2)
    ey, ex = random.randint(sy + 1, IND_ROWS), random.randint(sx + 1, IND_COLS)

    grid1[sy:ey, sx:ex], grid2[sy:ey, sx:ex] = grid2[sy:ey, sx:ex].copy(), grid1[sy:ey, sx:ex].copy()

    ind1[:], ind2[:] = grid1.flatten().tolist(), grid2.flatten().tolist()
    return ind1, ind2


def custom_select(pop, stagnation_counter, nd_select, k, forbidden_items=None):
    """
    Selects the next generation, dynamically adjusting strategies
    if the Stagnation Mode is active.

    @param pop: Current population list.
    @param stagnation_counter: Counter indicating how long the max score has been stagnant.
    @param nd_select: String defining the non-dominated sorting selection algorithm.
    @param k: Target number of individuals to select.
    @param forbidden_items: List of individuals (elites) to exclude from selection.
    @return: List of selected individuals for the next generation.
    """
    global STAGNATION_MODE, CURRENT_CXPB, CURRENT_MUTPB
    STAGNATION_MODE = (stagnation_counter >= STAGNATION_LIMIT)

    target_size = k
    seen = set()
    if forbidden_items and not STAGNATION_MODE:
        for f_ind in forbidden_items:
            seen.add(tuple(f_ind))

    select_size = int(len(pop) * 0.7)

    if STAGNATION_MODE:
        r = random.random()
        if r < 0.40:
            candidates = tools.selBest(pop, select_size)
        elif r < 0.50:
            candidates = tools.selRandom(pop, select_size)
        elif r < 0.65:
            candidates = tools.selNSGA2(pop, select_size, nd=nd_select)
        else:
            candidates = tools.selTournamentDCD(pop, select_size)
    else:
        candidates = tools.selTournamentDCD(pop, select_size)

    selected = []
    for ind in candidates:
        if tuple(ind) not in seen:
            seen.add(tuple(ind))
            selected.append(toolbox.clone(ind))
            if len(selected) == target_size:
                break

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
def directional_spread_mutation(grid):
    """
    Spreads existing digits to neighboring cells to extend paths.

    @param grid: 2D numpy array representing the individual.
    """
    indpb_r = 0.01 * random.randint(1, 5)
    for row in range(IND_ROWS):
        for col in range(IND_COLS):
            if random.random() < indpb_r:
                deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                possible = [(dr, dc) for dr, dc in deltas if 0 <= row + dr < IND_ROWS and 0 <= col + dc < IND_COLS]
                if possible:
                    dr, dc = random.choice(possible)
                    if (dr * dc != 0) != (random.random() < 0.1):
                        grid[row + dr, col + dc] = grid[row, col]
                    else:
                        grid[row + dr, col + dc] = random.choice([x for x in range(10) if x != grid[row, col]])


def cyclic_remapping_mutation(grid):
    """
    Shuffles a selected subset of digits, swapping their positions globally.

    @param grid: 2D numpy array representing the individual.
    """
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
            val = grid[i, j]
            if val in mapping:
                grid[i, j] = mapping[val]


def directional_cyclic_one_shift_mutation(grid):
    """
    Cyclically shifts a complete row, column, or diagonal line.

    @param grid: 2D numpy array representing the individual.
    """
    indpb_r = 0.01 * random.randint(1, 5)
    for row in range(IND_ROWS):
        for col in range(IND_COLS):
            if random.random() < indpb_r:
                deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                r, c = row, col
                dr, dc = random.choice(deltas)

                def get_line(r, c, dr, dc):
                    line = []
                    cr, cc = r - dr, c - dc
                    while 0 <= cr < IND_ROWS and 0 <= cc < IND_COLS:
                        line.insert(0, (cr, cc))
                        cr -= dr
                        cc -= dc
                    line.append((r, c))
                    cr, cc = r + dr, c + dc
                    while 0 <= cr < IND_ROWS and 0 <= cc < IND_COLS:
                        line.append((cr, cc))
                        cr += dr
                        cc += dc
                    return line

                line = get_line(r, c, dr, dc)
                if len(line) < 3:
                    r = random.randint(1, IND_ROWS - 2)
                    c = random.randint(1, IND_COLS - 2)
                    dr, dc = random.choice(deltas)
                    line = get_line(r, c, dr, dc)

                if len(line) < 3:
                    continue

                values = [grid[pr, pc] for pr, pc in line]
                shifted = [values[-1]] + values[:-1]

                for i, (pr, pc) in enumerate(line):
                    grid[pr, pc] = shifted[i]


def local_3x3_rotate_mutation(grid):
    """
    Rotates the 8 neighbors around a randomly selected center cell.

    @param grid: 2D numpy array representing the individual.
    """
    indpb_r = 0.01 * random.randint(1, 5)
    for row in range(1, IND_ROWS - 1):
        for col in range(1, IND_COLS - 1):
            if random.random() < indpb_r:
                deltas = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
                neighbors, positions = [], []

                for dr, dc in deltas:
                    nr, nc = row + dr, col + dc
                    neighbors.append(grid[nr, nc])
                    positions.append((nr, nc))

                if random.random() < 0.5:
                    shifted = [neighbors[-1]] + neighbors[:-1]
                else:
                    shifted = neighbors[1:] + [neighbors[0]]

                for i, (nr, nc) in enumerate(positions):
                    grid[nr, nc] = shifted[i]


def full_digit_cycle_mutation(grid):
    """
    Strong mutation: Shifts every digit globally (e.g., 0->3, 1->7).

    @param grid: 2D numpy array representing the individual.
    """
    digits = list(range(10))
    random.shuffle(digits)
    mapping = {digits[i]: digits[(i + 1) % 10] for i in range(10)}

    for i in range(IND_ROWS):
        for j in range(IND_COLS):
            grid[i, j] = mapping[grid[i, j]]


# =============================================================================
# 6. Mutation Config & Main Mutator
# =============================================================================
MUTATION_TYPES = [
    directional_spread_mutation,
    cyclic_remapping_mutation,
    directional_cyclic_one_shift_mutation,
    local_3x3_rotate_mutation,
    full_digit_cycle_mutation
]

NORMAL_PROBS = [0.30, 0.40, 0.05, 0.05, 0.20]
STAGNATION_PROBS = [0.05, 0.40, 0.05, 0.10, 0.40]


def custom_mutate(individual, indpb=0.05):
    """
    Applies a specific mutation strategy based on roulette wheel selection.

    @param individual: The genetic individual to mutate.
    @param indpb: Independent probability of mutation (unused directly here, kept for DEAP compatibility).
    @return: A tuple containing the mutated individual.
    """
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
    """
    Forces hardware-level randomness to inject extreme diversity into the run.
    """
    seed_bytes = os.urandom(4)
    new_seed = int.from_bytes(seed_bytes, byteorder='big')
    random.seed(new_seed)
    np.random.seed(new_seed)


def update_crowding(population):
    """
    Calculates and assigns crowding distance for NSGA-II selection.

    @param population: List of individuals.
    @return: A flattened list of individuals with updated crowding distances.
    """
    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    return [ind for front in fronts for ind in front]


def load_previous_best():
    """
    Loads historical top grids from the save file to initialize the population.

    @return: A tuple (protected_individual, loaded_individuals_list).
    """
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
    """
    Builds the initial generation by combining saved grids and random grids.

    @return: A list containing the initialized population.
    """
    protected, loaded = load_previous_best()
    pop = ([protected] if protected else []) + loaded

    if len(pop) < TARGET_POP:
        pop.extend(toolbox.population(n=TARGET_POP - len(pop)))

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return tools.selBest(pop, TARGET_POP)


def save_result(pop, g):
    """
    Outputs the final top individuals to the console and appends them to the file.

    @param pop: The final population list.
    @param g: The number of generations completed.
    """
    sys.stdout.write(f"\n{'=' * 60}\nEvolution finished after {g} generations.\nGENERATION SUCCESS\n")
    top_k = tools.selBest(pop, k=ELITE_SIZE * 4)

    with open(FILENAME, 'a') as f:
        f.write(f"\n--- Final TOP {len(top_k)} ---\n")
        for rank, ind in enumerate(top_k, 1):
            sys.stdout.write(f"Rank {rank} - Score: {ind.fitness.values[0]:.0f}, Count: {ind.fitness.values[1]:.0f}\n")
            for row in np.array(ind).reshape(IND_ROWS, IND_COLS):
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')

    sys.stdout.write(f"Final TOP saved to {FILENAME}\n")
    sys.stdout.flush()


# =============================================================================
# 8. Main Evolutionary Process
# =============================================================================
def numba_precomputation():
    """
    Triggers Numba JIT compilation before multiprocessing begins
    to prevent process locks and initialization overhead.
    """
    sys.stdout.write("Precompiling Numba function...\n")
    sys.stdout.flush()
    _has_path(np.zeros((IND_ROWS, IND_COLS), dtype=np.int64), np.array([1], dtype=np.int64))
    sys.stdout.write("Numba compilation complete.\n")
    sys.stdout.flush()


def analysis_file(pool, pop):
    """
    Evaluates the initially loaded population using a multiprocessing pool.

    @param pool: Multiprocessing pool instance.
    @param pop: The initial population list.
    @return: The global maximum score found in the initial population.
    """
    global GLOBAL_MAX_SCORE
    toolbox.register("map", pool.map)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    update_crowding(pop)

    GLOBAL_MAX_SCORE = max([ind.fitness.values[0] for ind in pop])
    sys.stdout.write(f"Initial Max Consecutive: {GLOBAL_MAX_SCORE:.0f}\n")
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
    sys.stdout.write(f"\n{'=' * 10} MASS MUTATION EVENT TRIGGERED {'=' * 10}\n")
    sys.stdout.write(f"Elite protection disabled. Forcing mutation on all {len(pop)} individuals.\n")
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

    sys.stdout.write(f"Mass Mutation complete → New Max: {new_max:.0f}\n{'=' * 51}\n")
    sys.stdout.flush()
    return new_max


def generation(g, pop, max_score_all_time, stagnation_counter, seed_counter, last_max):
    """
    Processes a single generation step. Stagnation and Seed counters are managed independently.

    @param g: Current generation index.
    @param pop: Current population list.
    @param max_score_all_time: Best score achieved globally across all generations.
    @param stagnation_counter: Counter for consecutive generations without improvement.
    @param seed_counter: Counter specifically tied to resetting the hardware random seed.
    @param last_max: Maximum score from the immediately preceding generation.
    @return: Updated tuple of (max_score_all_time, stagnation_counter, seed_counter, last_max).
    """
    global GLOBAL_MAX_SCORE

    # 1. Selection & Elite Preservation
    best_set = tools.selBest(pop, ELITE_BEST_SIZE)
    elites = list(map(toolbox.clone, best_set + tools.selNSGA2([i for i in pop if i not in best_set], ELITE_SIZE)))

    forbidden = elites

    # 2. Breed offsrping
    offspring_candidates = [p for p in tools.selBest(pop, len(pop))[ELITE_SIZE:]]
    offspring = list(map(toolbox.clone, toolbox.select(
        offspring_candidates, stagnation_counter, 'standard', TARGET_POP - ELITE_SIZE, forbidden_items=forbidden
    )))

    # 3. Crossover
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CURRENT_CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values

    # 4. Mutation
    for mutant in offspring:
        if random.random() < CURRENT_MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 5. Evaluation
    invalid = [ind for ind in pop + offspring if not ind.fitness.valid]
    if invalid:
        for ind, fit in zip(invalid, list(toolbox.map(toolbox.evaluate, invalid))):
            ind.fitness.values = fit

    pop[:] = elites + offspring
    update_crowding(pop)

    current_max = max([ind.fitness.values[0] for ind in pop])

    # 6. Record Breaking Check (Resets both counters on success)
    if current_max > max_score_all_time:
        sys.stdout.write(f"★ NEW RECORD! Gen {g}: {max_score_all_time:.0f} → {current_max:.0f}\n")
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

    # 7. Random Seed Diversification
    if seed_counter > G_STAG_GROUP:
        reset_random_seed()
        seed_counter = 0
        sys.stdout.flush()

    # 8. Console Logging
    if g % G_PRINT_GROUP == 0:
        all_scores = [ind.fitness.values[0] for ind in pop]
        current_avg = np.mean(all_scores)
        q1, median, q3 = np.percentile(all_scores, [25, 50, 75])

        sys.stdout.write(
            f"Gen {g:5d} | Max: {current_max:>6.0f} | Avg: {current_avg:>6.1f} | "
            f"Q3: {q3:>6.1f} | Med: {median:>6.1f} | Q1: {q1:>6.1f} \n"
        )
        sys.stdout.flush()

    # 9. Mass Mutation Check
    if stagnation_counter >= STAGNATION_LIMIT:
        new_max = perform_mass_mutation(pop)
        stagnation_counter = 0
        last_max = new_max
        max_score_all_time = new_max

    return max_score_all_time, stagnation_counter, seed_counter, last_max


# =============================================================================
# 9. Application Entry Point
# =============================================================================
def main():
    numba_precomputation()
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