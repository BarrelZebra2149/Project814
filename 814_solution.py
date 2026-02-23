import random
import numpy as np
from collections import deque
import sys
import os
import multiprocessing
import random
from collections import Counter

from deap import base
from deap import creator
from deap import tools
from scoop import futures

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
toolbox = base.Toolbox()


def count_occurrences(grid, n):
    S = str(n)
    target_len = len(S)
    digits = [int(d) for d in S]
    count = 0
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def is_valid(r, c):
        return 0 <= r < IND_ROWS and 0 <= c < IND_COLS

    def dfs(r, c, idx):
        nonlocal count
        if idx == target_len - 1:
            count += 1
            return

        if count > 500: return

        next_digit = digits[idx + 1]
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == next_digit:
                dfs(nr, nc, idx + 1)

    starts = [(r, c) for r in range(IND_ROWS) for c in range(IND_COLS) if grid[r][c] == digits[0]]
    if not starts: return 0
    if target_len == 1: return 1

    for r, c in starts:
        dfs(r, c, 0)
        if count > 500: return count
    return count


def eval_814_heuristic(individual):
    grid = np.array(individual).reshape(IND_ROWS, IND_COLS)
    total_heuristic = 0.0
    n = 1
    MAX_N = 50000

    while n < MAX_N:
        M = count_occurrences(grid, n)
        if M == 0:
            X = n - 1
            break

        if n < 10:
            total_heuristic += 1.0
        else:
            total_heuristic += 1.0 / M

        n += 1
    else:
        X = MAX_N - 1

    return X, total_heuristic


def custom_mutate(individual, indpb=0.17):
    digit_counts = Counter(individual)

    if len(digit_counts) < 2:
        return individual,

    max_digit = max(digit_counts, key=digit_counts.get)
    max_count = digit_counts[max_digit]

    other_digits = [d for d in range(10) if d != max_digit]
    if not other_digits:
        return individual,
    total_other = sum(digit_counts.get(d, 0) for d in other_digits)
    avg_other = total_other / len(other_digits)

    rand_sub = random.randint(2, 3)
    change_count = int(max_count - avg_other - rand_sub)
    if change_count <= 0:
        return individual,

    locations = [i for i, d in enumerate(individual) if d == max_digit]
    if not locations:
        return individual,

    if random.random() >= indpb:
        return individual,

    change_count = min(change_count, len(locations))

    random.shuffle(locations)
    to_change = locations[:change_count]

    below_avg_digits = [d for d in other_digits if digit_counts.get(d, 0) < avg_other]

    for pos in to_change:
        if below_avg_digits:
            new_digit = random.choice(below_avg_digits)
        else:
            new_digit = random.choice(other_digits)

        individual[pos] = new_digit

        digit_counts[max_digit] -= 1
        digit_counts[new_digit] += 1

    return individual,


toolbox.register("evaluate", eval_814_heuristic)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.17)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def main():
    loaded = []
    protected = None
    if os.path.exists('best_output_double.txt'):
        with open('best_output.txt', 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    ind_list = list(map(int, line.split(',')))
                    if len(ind_list) == IND_SIZE:
                        ind = creator.Individual(ind_list)
                        if idx == 0:
                            protected = ind
                        else:
                            loaded.append(ind)
                except ValueError:
                    pass
    pop = []
    if protected is not None:
        pop.append(protected)
    pop.extend(loaded)
    target_pop_size = 300
    remaining = target_pop_size - len(pop)
    if remaining > 0:
        pop.extend(toolbox.population(n=remaining))
    if len(pop) > target_pop_size:
        pop = pop[:target_pop_size]
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        CXPB, MUTPB = 0.3, 0.5
        g = 0
        while g < 50000:
            g += 1
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]

            if g % 100 == 0:
                length = len(pop)
                mean = sum(fits) / length
                sys.stdout.write(f"-- Generation {g} --\n")
                sys.stdout.write(f" Max Score: {max(fits):.4f}\n")
                sys.stdout.write(f" Avg Score: {mean:.4f}\n")
        best_ind = tools.selBest(pop, 1)[0]
        sys.stdout.write(f"\nBest individual Score: {best_ind.fitness.values[0]:.4f}\n")
        grid = np.array(best_ind).reshape(IND_ROWS, IND_COLS)
        for row in grid:
            sys.stdout.write(''.join(map(str, row)) + '\n')
        flat = list(best_ind)
        sys.stdout.write("\nDigit Counts:\n")
        for i in range(10):
            sys.stdout.write(f"{i}: {flat.count(i)} ")
        sys.stdout.write("\n")
        with open('best_output_double.txt', 'a') as f:
            f.write(','.join(map(str, best_ind)) + '\n')


if __name__ == "__main__":
    main()


