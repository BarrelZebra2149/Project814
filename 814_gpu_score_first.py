import random
import numpy as np
import sys
import os
import multiprocessing
from deap import base, creator, tools
from numba import njit

# --- Constants (merged A + B) ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = 'best_output_double.txt'

TARGET_POP = 500                    # from A
NGEN = 30000
G_PRINT_GROUP = 250             # balanced frequency
G_LOCAL_THRESHOLD = 2000
STAGNATION_LIMIT = 10
G_BOOM_THRESHOLD = 2000
ELITE_SIZE = 10

# Dynamic CXPB/MUTPB thresholds (array for readability)
PHASES = [
    (4000,  0.25, 0.60),   # gen < 4000: low crossover, high mutation (exploration)
    (12000, 0.45, 0.35),   # gen < 12000: moderate
    (NGEN,  0.65, 0.15),   # gen >= 12000: high crossover, low mutation (exploitation)
]

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# ====================== MIGRATED FROM A ======================
def custom_select(pop, stagnation_counter, nd_select, k):
    target_size = k
    select_size = int(len(pop) * 0.7)
    if stagnation_counter >= STAGNATION_LIMIT:
        r = random.random()
        if r < 0.25:
            candidates = tools.selWorst(pop, select_size)  # oversample for uniqueness
        elif r < 0.50:
            candidates = tools.selRandom(pop, select_size)
        elif r < 0.75:
            candidates = tools.selNSGA2(pop, select_size, nd=nd_select)
        else:
            candidates = tools.selTournamentDCD(pop, select_size)
    else:
        candidates = tools.selBest(pop, select_size)

    # Unique filtering: remove duplicates, fill with random if short
    seen = set()
    selected = []
    for ind in candidates:
        ind_tuple = tuple(ind)
        if ind_tuple not in seen:
            seen.add(ind_tuple)
            selected.append(toolbox.clone(ind))
            if len(selected) == target_size:
                break

    # If short, fill with random new individuals
    while len(selected) < target_size:
        new_ind = toolbox.individual()
        selected.append(new_ind)

    return selected[:target_size]


def custom_mate(ind1, ind2):
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)
    sy = random.randint(0, IND_ROWS - 2)
    sx = random.randint(0, IND_COLS - 2)
    ey = random.randint(sy + 1, IND_ROWS)
    ex = random.randint(sx + 1, IND_COLS)
    grid1[sy:ey, sx:ex], grid2[sy:ey, sx:ex] = grid2[sy:ey, sx:ex].copy(), grid1[sy:ey, sx:ex].copy()
    ind1[:], ind2[:] = grid1.flatten().tolist(), grid2.flatten().tolist()
    return ind1, ind2


def custom_mutate(individual):
    if random.random() < 0.5:
        tools.mutUniformInt(individual, 0, 9, 0.05)
    else:
        a, b = random.sample(range(10), 2)
        for i in range(len(individual)):
            if individual[i] == a:
                individual[i] = b
            elif individual[i] == b:
                individual[i] = a
    return individual,


def update_crowding(population):
    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    assign = tools.emo.assignCrowdingDist
    for front in fronts:
        assign(front)
    return [ind for front in fronts for ind in front]



# ====================== Numba-accelerated path check ======================
@njit
def _has_path(grid: np.ndarray, digits: np.ndarray) -> bool:
    rows = grid.shape[0]
    cols = grid.shape[1]
    digit_len = digits.shape[0]
    if digit_len == 0:
        return False

    deltas = np.array([[-1, -1], [-1, 0], [-1, 1],
                       [0, -1],           [0, 1],
                       [1, -1],  [1, 0],  [1, 1]], dtype=np.int64)

    stack = []
    first_digit = digits[0]

    # Find all starting positions
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == first_digit:
                stack.append((r, c, 0))

    while len(stack) > 0:
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


# ====================== Updated count_occurrences ======================
def count_occurrences(grid, n):
    if n <= 0:
        return 0
    digits_list = [int(d) for d in str(n)]
    digits = np.array(digits_list, dtype=np.int64)
    return 1 if _has_path(grid, digits) else 0


# ====================== Updated eval_814_heuristic (grid dtype) ======================
def eval_814_heuristic(individual):
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

# ====================== Toolbox Registration (A style) ======================
toolbox.register("evaluate", eval_814_heuristic)
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", custom_select)
toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def load_previous_best():
    # ... (same as original Code B - protected + loaded) ...
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


# ====================== Updated main() with sys.stdout.write + flush ======================
def main():
    sys.stdout.write("Precompiling Numba function...\n")
    sys.stdout.flush()
    dummy_grid = np.zeros((IND_ROWS, IND_COLS), dtype=np.int64)
    dummy_digits = np.array([1], dtype=np.int64)
    _has_path(dummy_grid, dummy_digits)
    sys.stdout.write("Numba compilation complete.\n")
    sys.stdout.flush()

    protected, loaded = load_previous_best()
    pop = []
    if protected: pop.append(protected)
    pop.extend(loaded)
    if len(pop) < TARGET_POP:
        pop.extend(toolbox.population(n=TARGET_POP - len(pop)))
        pop = pop[:TARGET_POP]
    else:
        pop = random.sample(pop, TARGET_POP)

    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)

        # Initial evaluation
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        update_crowding(pop)

        max_score_all_time = max([ind.fitness.values[0] for ind in pop]) if pop else -1.0
        sys.stdout.write(f"Initial Max Score: {max_score_all_time:.0f}\n")
        sys.stdout.flush()

        best_ind = None
        local_max_ind = None
        improved = False
        passed = True
        stagnation_counter = 0
        last_max = max_score_all_time

        for g in range(1, NGEN + 1):
            cxpb, mutpb = next(((c, m) for gen, c, m in PHASES if g < gen), (0.65, 0.15))
            elites = tools.selBest(pop, ELITE_SIZE)
            elites = list(map(toolbox.clone, elites))

            n_offspring = TARGET_POP - ELITE_SIZE
            nd_method = 'standard' if NGEN > 15000 else 'log'
            offspring = custom_select(pop, stagnation_counter, nd_method, n_offspring)
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            pop[:] = elites + offspring
            update_crowding(pop)


            current_max = max([ind.fitness.values[0] for ind in pop])

            if current_max > max_score_all_time:
                sys.stdout.write(f"★ NEW RECORD! Gen {g}: {max_score_all_time:.0f} → {current_max:.0f}\n")
                sys.stdout.flush()
                max_score_all_time = current_max
                best_ind = toolbox.clone(pop[np.argmax([ind.fitness.values[0] for ind in pop])])
                improved = True
                stagnation_counter = 0
                last_max = current_max
            else:
                stagnation_counter += 1 if current_max <= last_max else 0
                last_max = current_max

            if g % G_PRINT_GROUP == 0:
                current_avg = sum([ind.fitness.values[0] for ind in pop]) / len(pop)
                sys.stdout.write(f"Gen {g:5d} | Current Max: {current_max:>6.0f} | "
                                 f"All-time Max: {max_score_all_time:>6.0f} | Avg: {current_avg:>6.2f}\n")
                sys.stdout.flush()

        sys.stdout.write("\n" + "=" * 60 + "\n")
        sys.stdout.write(f"Evolution finished.\n")
        sys.stdout.flush()

        saved_ind = best_ind if (best_ind and improved) else local_max_ind
        if saved_ind:
            grid = np.array(saved_ind).reshape(IND_ROWS, IND_COLS)
            with open(FILENAME, 'a') as f:
                for row in grid:
                    f.write(''.join(map(str, row)) + '\n')
                f.write('\n')
            print(f"Final Saved (Form: {saved_ind.fitness.values[0]:.0f}, Score: {saved_ind.fitness.values[1]:.0f})")
        else:
            print("\nNO PROGRESS OR NO CANDIDATE TO SAVE!")

        if passed:
            print("\nGENERATION SUCCESS")
            top_k = tools.selBest(pop, k=ELITE_SIZE*3)
            with open(FILENAME, 'a') as f:
                sys.stdout.write(f"\n--- Final TOP {ELITE_SIZE*3} after {g} generations ---\n")
                for rank, ind in enumerate(top_k, 1):
                    grid = np.array(ind).reshape(IND_ROWS, IND_COLS)
                    sys.stdout.write(f"Rank {rank} - Form: {ind.fitness.values[0]:.0f}, Score: {ind.fitness.values[1]:.0f}\n")
                    for row in grid:
                        f.write(''.join(map(str, row)) + '\n')
                    f.write('\n')
            sys.stdout.write(f"Final TOP {ELITE_SIZE*3} saved to {FILENAME}\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("No improvement - nothing saved.\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()