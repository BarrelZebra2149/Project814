import random
import numpy as np
import sys
import os
import multiprocessing
from collections import Counter
from deap import base, creator, tools

# --- Setup ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = 'best_output_double.txt'

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# --- Core Logic ---
def count_occurrences(grid, n):
    S = str(n)
    target_len = len(S)
    digits = [int(d) for d in S]
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def is_valid(r, c):
        return 0 <= r < IND_ROWS and 0 <= c < IND_COLS

    found = False

    def dfs(r, c, idx):
        nonlocal found
        if found: return
        if idx == target_len - 1:
            found = True
            return

        next_digit = digits[idx + 1]
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == next_digit:
                dfs(nr, nc, idx + 1)
            if found: return

    starts = [(r, c) for r in range(IND_ROWS) for c in range(IND_COLS) if grid[r][c] == digits[0]]
    if not starts: return 0

    for r, c in starts:
        dfs(r, c, 0)
        if found: break

    return 1 if found else 0


def eval_814_heuristic(individual):
    grid = np.array(individual).reshape(IND_ROWS, IND_COLS)
    MAX_N = 50000

    found_set = set()  # Tracks all discovered numbers (forward + valid reverses)

    # Calculate consecutive score (1+) with reverse optimization
    n = 1
    while n < MAX_N:
        # Compute reverse (for 1-digit, reverse is itself)
        rev_str = str(n)[::-1]
        rev_n = int(rev_str)

        # If forward or reverse already discovered, consider it formable

        if n in found_set or (n % 10 != 0 and rev_n in found_set):
            n += 1
            continue

        # Otherwise, perform DFS to check existence
        if count_occurrences(grid, n):
            found_set.add(n)
            found_set.add(rev_n)  # Add reverse too (free discovery)
        else:
            current_score = n - 1
            break
        n += 1
    else:
        current_score = MAX_N - 1

    # Calculate formable count (1000-9999) using the same found_set
    formable_count = max(0, min(9999, current_score) - 1000 + 1)

    start_num = max(1000, current_score + 1)
    for num in range(start_num, 10000):
        rev_str = str(num)[::-1]
        rev_num = int(rev_str)

        # If forward or reverse already in set, count it (skip DFS)
        if num in found_set or (n % 10 != 0 and rev_num in found_set):
            formable_count += 1
            continue

        # Perform DFS only if neither is known
        if count_occurrences(grid, num):
            formable_count += 1
            found_set.add(num)
            found_set.add(rev_num)

    # Prioritize current_score: return it as the primary objective
    return float(current_score), float(formable_count)


def custom_mate(ind1, ind2):
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)

    s_y = random.randint(0, IND_ROWS - 1)
    e_y = random.randint(s_y + 1, IND_ROWS)
    s_x = random.randint(0, IND_COLS - 1)
    e_x = random.randint(s_x + 1, IND_COLS)

    # Swap rectangular block
    temp = grid1[s_y:e_y, s_x:e_x].copy()
    grid1[s_y:e_y, s_x:e_x] = grid2[s_y:e_y, s_x:e_x]
    grid2[s_y:e_y, s_x:e_x] = temp

    ind1[:] = grid1.ravel().tolist()
    ind2[:] = grid2.ravel().tolist()

    return ind1, ind2


# ------------------------------------------------
#   Custom Mutation: 50% uniform int, 50% full digit swap
# ------------------------------------------------
def custom_mutate(individual, indpb=0.05):
    """
    When mutation is triggered:
      - 50% chance: standard uniform replacement on some positions
      - 50% chance: swap ALL occurrences of two randomly chosen different digits
    """
    if random.random() < 0.5:
        # Standard DEAP uniform integer mutation
        tools.mutUniformInt(individual, low=0, up=9, indpb=indpb)
    else:
        # Full digit swap: choose two different digits A and B, swap all of them
        digits = list(range(10))
        random.shuffle(digits)
        a, b = digits[0], digits[1]  # two different digits

        # Swap every occurrence of a ? b in the flat list
        for i in range(len(individual)):
            if individual[i] == a:
                individual[i] = b
            elif individual[i] == b:
                individual[i] = a

    return individual,


# --- GA Registration ---
toolbox.register("evaluate", eval_814_heuristic)
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate, indpb=0.05)
# -----------------------------
# Crowding distance (robust)
# -----------------------------
def _assign_crowding_dist_fallback(front):
    """Fallback crowding distance assignment for a single front (NSGA-II style)."""
    if not front:
        return
    if len(front) <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    for ind in front:
        ind.crowding_dist = 0.0

    nobj = len(front[0].fitness.values)
    for m in range(nobj):
        front.sort(key=lambda ind: ind.fitness.values[m])
        front[0].crowding_dist = float("inf")
        front[-1].crowding_dist = float("inf")
        fmin = front[0].fitness.values[m]
        fmax = front[-1].fitness.values[m]
        denom = fmax - fmin
        if denom == 0:
            continue
        for i in range(1, len(front) - 1):
            prev_f = front[i - 1].fitness.values[m]
            next_f = front[i + 1].fitness.values[m]
            front[i].crowding_dist += (next_f - prev_f) / denom


def update_crowding(population):
    """
    Assign crowding distance to every individual in the population.
    Works across DEAP versions (tools.emo.assignCrowdingDist vs missing symbol).
    """
    fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)

    # Try DEAP's implementation if available (common location: tools.emo.assignCrowdingDist)
    assign = None
    if hasattr(tools, "emo") and hasattr(tools.emo, "assignCrowdingDist"):
        assign = tools.emo.assignCrowdingDist

    for front in fronts:
        if assign is not None:
            assign(front)
        else:
            _assign_crowding_dist_fallback(front)


# Use selTournamentDCD with tournsize=2 (standard for dominance + crowding)
toolbox.register("select", tools.selTournamentDCD)

toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


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


def main():
    protected, loaded = load_previous_best()

    pop = []
    if protected: pop.append(protected)
    pop.extend(loaded)

    TARGET_POP = 300
    if len(pop) < TARGET_POP:
        pop.extend(toolbox.population(n=TARGET_POP - len(pop)))
    pop = pop[:TARGET_POP]

    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)

        # Initial fitness
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        update_crowding(pop)

        CXPB, MUTPB = 0.5, 0.2
        NGEN = 10000
        best_current_score = 0.0
        best_ind = None
        for g in range(1, NGEN + 1):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            update_crowding(pop)
            pop_max_v = [ind.fitness.values[0] for ind in pop]
            current_max = max(pop_max_v)
            if current_max > best_current_score:
                best_current_score = current_max
                best_ind = max(pop, key=lambda ind: ind.fitness.values[0])
            # Stats Output
            if g % 500 == 0:
                formable = [ind.fitness.values[1] for ind in pop]
                sys.stdout.write(f"--Generation {g}--\n")
                sys.stdout.write(
                    f"Current Score (1+)\tMax: {current_max:>7.0f}\tAvg: {sum(pop_max_v) / len(pop):>7.1f}\n")
                sys.stdout.write(
                    f"Formable Count\tMax: {max(formable):>7.0f}\tAvg: {sum(formable) / len(pop):>7.1f}\n")
        print("\n" + "=" * 60)
        print(f"Evolution finished after {NGEN} generations")
        
        if best_ind is not None:
            print("\n" + "=" * 80)
            print("GRID WITH THE HIGHEST CURRENT SCORE (consecutive from 1)")
            print("=" * 80)
            print(f"Current Score : {best_current_score:.0f}")
            print(f"Formable Count: {best_ind.fitness.values[1]:.0f}")
            grid = np.array(best_ind).reshape(IND_ROWS, IND_COLS)
            for row in grid:
                print(''.join(map(str, row)))
            flat = list(best_ind)
            counts = Counter(flat)
            print("\nDigit distribution:", ' '.join(f"{d}:{counts[d]}" for d in range(10)))

            # Save to file (appended as 8 lines + blank line)
            with open(FILENAME, 'a') as f:
                for row in grid:
                    f.write(''.join(map(str, row)) + '\n')
                f.write('\n')
            print("Highest current_score grid saved to file.")
        else:
            print("No valid individuals found to save.")  
            
        top3 = tools.selBest(pop, k=3)
        for rank, ind in enumerate(top3, 1):
            curr, form = ind.fitness.values  # primary first
            print(f"\nTop #{rank} Individual:")
            print(f" Current consecutive score : {curr:>7.0f}")
            print(f" Formable count (1000-9999): {form:>7.0f}")

            grid = np.array(ind).reshape(IND_ROWS, IND_COLS)
            print(" Grid:")
            for row in grid:
                print(' ' + ''.join(map(str, row)))

            flat = list(ind)
            counts = Counter(flat)
            print(" Digit distribution:", ' '.join(f"{d}:{counts[d]}" for d in range(10)))
            print("-" * 50)

        # Save Best
        with open(FILENAME, 'a') as f:
            grid = np.array(top3[0]).reshape(IND_ROWS, IND_COLS)
            for row in grid:
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')


if __name__ == "__main__":
    main()
