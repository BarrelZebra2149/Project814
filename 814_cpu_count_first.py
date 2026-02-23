import random
import numpy as np
import sys
import os
import multiprocessing
from deap import base, creator, tools
from numba import njit

# --- Constants ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = 'best_output_double_final.txt'
TARGET_POP = 500
NGEN = 10000
G_PRINT_GROUP = 10
G_LOCAL_THRESHOLD = 2000
STAGNATION_LIMIT = 300
G_BOOM_THRESHOLD = 2000
ELITE_SIZE = 10
# Dynamic crossover/mutation probabilities
CURRENT_CXPB = 0.5
CURRENT_MUTPB = 0.3
# Global max score for Missing Hunter
GLOBAL_MAX_SCORE = 0.0
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# ====================== STAGNATION CONTROL ======================
STAGNATION_MODE = False


# ====================== Custom Selection ======================
def custom_select(pop, stagnation_counter, nd_select, k, forbidden_items=None):
    """Select parents with special handling for stagnation"""
    global STAGNATION_MODE, CURRENT_CXPB, CURRENT_MUTPB

    STAGNATION_MODE = (stagnation_counter >= STAGNATION_LIMIT)

    if STAGNATION_MODE:
        CURRENT_CXPB = 0.30
        CURRENT_MUTPB = 0.50
    else:
        CURRENT_CXPB = 0.50
        CURRENT_MUTPB = 0.30

    target_size = k
    seen = set()

    if forbidden_items:
        for f_ind in forbidden_items:
            seen.add(tuple(f_ind))

    select_size = int(len(pop) * 0.7)

    if STAGNATION_MODE:
        r = random.random()
        if r < 0.2:
            candidates = tools.selBest(pop, select_size)
        elif r < 0.3:
            candidates = tools.selRandom(pop, select_size)
        elif r < 0.6:
            candidates = tools.selNSGA2(pop, select_size, nd=nd_select)
        else:
            candidates = tools.selTournamentDCD(pop, select_size)
    else:
        candidates = tools.selTournamentDCD(pop, select_size)

    selected = []
    for ind in candidates:
        ind_tuple = tuple(ind)
        if ind_tuple not in seen:
            seen.add(ind_tuple)
            selected.append(toolbox.clone(ind))
            if len(selected) == target_size:
                break

    if len(selected) < target_size:
        fill_count = target_size - len(selected)
        fillers = tools.selNSGA2(pop, fill_count)
        for f in fillers:
            child = toolbox.clone(f)
            if random.random() < 0.2:
                toolbox.mutate(child)
                del child.fitness.values
            selected.append(child)

    return selected[:target_size]


# ====================== Genetic Operators ======================
def custom_mate(ind1, ind2):
    """Crossover: swap random rectangular subgrid"""
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)
    sy = random.randint(0, IND_ROWS - 2)
    sx = random.randint(0, IND_COLS - 2)
    ey = random.randint(sy + 1, IND_ROWS)
    ex = random.randint(sx + 1, IND_COLS)
    grid1[sy:ey, sx:ex], grid2[sy:ey, sx:ex] = grid2[sy:ey, sx:ex].copy(), grid1[sy:ey, sx:ex].copy()
    ind1[:], ind2[:] = grid1.flatten().tolist(), grid2.flatten().tolist()
    return ind1, ind2


# ====================== Mutation Types ======================
def structural_mutation(grid):
    """Structural: sparse neighbor swaps"""
    indpb_r = 0.01 * random.randint(1, 5)
    for row in range(IND_ROWS):
        for col in range(IND_COLS):
            if random.random() < indpb_r:
                deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                valid = [(row + dr, col + dc) for dr, dc in deltas
                         if 0 <= row + dr < IND_ROWS and 0 <= col + dc < IND_COLS]
                if valid:
                    nr, nc = random.choice(valid)
                    if grid[row, col] != grid[nr, nc]:
                        grid[row, col], grid[nr, nc] = grid[nr, nc], grid[row, col]


def macro_mutation(grid):
    """Macro: swap two entire digit types"""
    a, b = random.sample(range(10), 2)
    m_a = (grid == a)
    m_b = (grid == b)
    grid[m_a] = b
    grid[m_b] = a


def structural_mutation_direction(grid):
    """Micro: single-cell directional change"""
    indpb_r = 0.01 * random.randint(1, 5)
    for row in range(IND_ROWS):
        for col in range(IND_COLS):
            if random.random() < indpb_r:
                center_val = grid[row, col]
                deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                possible_dirs = [(dr, dc) for dr, dc in deltas
                                 if 0 <= row + dr < IND_ROWS and 0 <= col + dc < IND_COLS]
                if possible_dirs:
                    dr, dc = random.choice(possible_dirs)
                    nr, nc = row + dr, col + dc
                    is_reverse = (random.random() < 0.1)
                    is_diagonal = (dr * dc != 0)
                    if is_diagonal != is_reverse:
                        grid[nr, nc] = center_val
                    else:
                        grid[nr, nc] = random.choice([x for x in range(10) if x != center_val])


def cataclysmic_mutation(grid):
    """Cataclysmic: full permutation of 0-9"""
    perm = np.random.permutation(10)
    grid[:] = perm[grid]


def missing_hunter(grid):
    """Missing Hunter: inserts high score number"""
    global GLOBAL_MAX_SCORE
    base = max(int(GLOBAL_MAX_SCORE), 1000)
    offset = random.randint(5, 180)
    target = base + offset
    digits = [int(d) for d in str(target)]

    row = random.randint(0, IND_ROWS - 1)
    col = random.randint(0, IND_COLS - 1)
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dr, dc = random.choice(deltas)

    for i, d in enumerate(digits):
        nr = row + i * dr
        nc = col + i * dc
        if 0 <= nr < IND_ROWS and 0 <= nc < IND_COLS:
            grid[nr, nc] = d
        else:
            break


# Mutation type array (order matters)
MUTATION_TYPES = [
    structural_mutation,
    macro_mutation,
    structural_mutation_direction,
    cataclysmic_mutation,
    missing_hunter
]
# Probability distributions
NORMAL_PROBS = [0.30, 0.20, 0.30, 0.10, 0.10]
STAGNATION_PROBS = [0.10, 0.15, 0.30, 0.10, 0.35]


# ====================== Custom Mutate ======================
def custom_mutate(individual, indpb=0.05):
    """Main mutation entry: uses different probabilities based on stagnation"""
    global STAGNATION_MODE
    r = random.random()
    grid = np.array(individual).reshape(IND_ROWS, IND_COLS)

    probs = STAGNATION_PROBS if STAGNATION_MODE else NORMAL_PROBS

    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r < cum:
            MUTATION_TYPES[i](grid)
            break

    individual[:] = grid.ravel().tolist()
    return individual,


# ====================== Evaluation & Utilities ======================
def update_crowding(population):
    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    assign = tools.emo.assignCrowdingDist
    for front in fronts:
        assign(front)
    return [ind for front in fronts for ind in front]


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


def eval_814_heuristic(individual):
    global GLOBAL_MAX_SCORE
    grid = np.array(individual, dtype=np.int64).reshape(IND_ROWS, IND_COLS)
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

    # Update global max score
    if current_score > GLOBAL_MAX_SCORE:
        GLOBAL_MAX_SCORE = current_score

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


def load_previous_best():
    loaded = []
    protected = None
    if os.path.exists(FILENAME):
        with open(FILENAME, 'r') as f:
            valid_lines = [line.strip() for line in f if len(line.strip()) == 14 and line.strip().isdigit()]
        for block_start in range(0, len(valid_lines), 8):
            if block_start + 7 >= len(valid_lines): break
            block = valid_lines[block_start:block_start + 8]
            ind_list = []
            for row in block:
                ind_list.extend(int(d) for d in row)
            if len(ind_list) == IND_SIZE:
                ind = creator.Individual(ind_list)
                if block_start == 0:
                    protected = ind
                else:
                    loaded.append(ind)
    return protected, loaded


toolbox.register("evaluate", eval_814_heuristic)
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate, indpb=0.05)
toolbox.register("select", custom_select)
toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ====================== MAIN EVOLUTION LOOP ======================
def main():
    global STAGNATION_MODE, GLOBAL_MAX_SCORE
    """Main GA loop with stagnation handling"""

    sys.stdout.write("Precompiling Numba function...\n")
    sys.stdout.flush()
    dummy_grid = np.zeros((IND_ROWS, IND_COLS), dtype=np.int64)
    dummy_digits = np.array([1], dtype=np.int64)
    _has_path(dummy_grid, dummy_digits)
    sys.stdout.write("Numba compilation complete.\n")
    sys.stdout.flush()

    protected, loaded = load_previous_best()
    pop = []
    if protected:
        pop.append(protected)
    pop.extend(loaded)

    if len(pop) < TARGET_POP:
        pop.extend(toolbox.population(n=TARGET_POP - len(pop)))
        pop = pop[:TARGET_POP]
    else:
        pop = random.sample(pop, TARGET_POP)

    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)

        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        update_crowding(pop)

        max_score_all_time = max([ind.fitness.values[0] for ind in pop]) if pop else -1.0
        GLOBAL_MAX_SCORE = max_score_all_time
        sys.stdout.write(f"Initial Max Consecutive: {max_score_all_time:.0f}\n")
        sys.stdout.flush()

        local_max_score = -1.0
        passed = True
        stagnation_counter = 0
        last_max = max_score_all_time

        for g in range(1, NGEN + 1):
            cxpb = CURRENT_CXPB
            mutpb = CURRENT_MUTPB

            best_one = tools.selBest(pop, 1)
            remaining_pool = [ind for ind in pop if ind not in best_one]
            nsga_rest = tools.selNSGA2(remaining_pool, ELITE_SIZE - 1)
            elites = list(map(toolbox.clone, best_one + nsga_rest))

            if STAGNATION_MODE:
                forbidden_items = None
            else:
                forbidden_items = elites

            n_offspring = TARGET_POP - ELITE_SIZE
            sorted_pop = tools.selBest(pop, len(pop))

            if STAGNATION_MODE:
                selection_pool = sorted_pop[ELITE_SIZE:]
            else:
                selection_pool = sorted_pop
                n_offspring += ELITE_SIZE

            nd_method = 'standard' if NGEN > 15000 else 'log'

            offspring = toolbox.select(selection_pool, stagnation_counter, nd_method, n_offspring,
                                       forbidden_items=forbidden_items)
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop[:] = elites + offspring

            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            update_crowding(pop)

            current_max = max([ind.fitness.values[0] for ind in pop])

            if current_max > max_score_all_time:
                sys.stdout.write(f"★ NEW RECORD! Gen {g}: {max_score_all_time:.0f} → {current_max:.0f}\n")
                sys.stdout.flush()
                max_score_all_time = current_max
                GLOBAL_MAX_SCORE = current_max
                best_idx = np.argmax([ind.fitness.values[0] for ind in pop])
                best_ind = toolbox.clone(pop[best_idx])

                try:
                    with open(FILENAME, 'a') as f:
                        grid_record = np.array(best_ind).reshape(IND_ROWS, IND_COLS)
                        for row in grid_record:
                            f.write(''.join(map(str, row)) + '\n')
                        f.write('\n')
                except Exception as e:
                    sys.stdout.write(f" >> Error saving record: {e}\n")

                stagnation_counter = 0
                STAGNATION_MODE = False
                last_max = current_max
            else:
                stagnation_counter += 1 if current_max <= last_max else 0
                last_max = current_max

            if g % G_PRINT_GROUP == 0:
                all_scores = [ind.fitness.values[0] for ind in pop]

                current_avg = np.mean(all_scores)
                q1, median, q3 = np.percentile(all_scores, [25, 50, 75])

                sys.stdout.write(
                    f"Gen {g:5d} | Max: {current_max:>6.0f} | "
                    f"Avg: {current_avg:>6.1f} | global_max : {max_score_all_time:>6.0f} | "
                    f"Q3(75%): {q3:>6.1f} | Med(50%): {median:>6.1f} | Q1(25%): {q1:>6.1f} | "
                    f"Stagnation: {STAGNATION_MODE}\n"
                )
                sys.stdout.flush()

        sys.stdout.write("\n" + "=" * 60 + "\n")
        sys.stdout.write(f"Evolution finished after {NGEN} generations.\n")
        sys.stdout.flush()

        if passed:
            print("\nGENERATION SUCCESS")
            top_k = tools.selBest(pop, k=ELITE_SIZE * 3)
            with open(FILENAME, 'a') as f:
                sys.stdout.write(f"\n--- Final TOP {ELITE_SIZE * 3} after {g} generations ---\n")
                for rank, ind in enumerate(top_k, 1):
                    grid = np.array(ind).reshape(IND_ROWS, IND_COLS)
                    sys.stdout.write(
                        f"Rank {rank} - Consecutive: {ind.fitness.values[0]:.0f}, Density: {ind.fitness.values[1]:.0f}\n")
                    for row in grid:
                        f.write(''.join(map(str, row)) + '\n')
                    f.write('\n')
            sys.stdout.write(f"Final TOP {ELITE_SIZE * 3} saved to {FILENAME}\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("No improvement - nothing saved.\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
