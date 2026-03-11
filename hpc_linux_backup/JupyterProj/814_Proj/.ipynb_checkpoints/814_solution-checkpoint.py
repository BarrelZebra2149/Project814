import random
import numpy as np
from collections import deque
import sys
import os

from deap import base
from deap import creator
from deap import tools
from scoop import futures

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 8 * 14

INT_MIN, INT_MAX = 0, 9
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval_814_num(individual):
    grid = np.array(individual).reshape(8, 14)

    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def is_valid(r, c):
        return 0 <= r < 8 and 0 <= c < 14

    def can_form(n):
        S = str(n)
        l = len(S)
        dig = [int(d) for d in S]

        q = deque()
        visited = set()
        has_start = False

        for r in range(8):
            for c in range(14):
                if grid[r, c] == dig[0]:
                    has_start = True
                    state = (1, r, c)
                    if state not in visited:
                        visited.add(state)
                        q.append((r, c, 1))

        if l == 1:
            return has_start

        while q:
            cr, cc, cpos = q.popleft()
            if cpos == l:
                return True
            for dr, dc in deltas:
                nr = cr + dr
                nc = cc + dc
                if is_valid(nr, nc) and grid[nr, nc] == dig[cpos]:
                    new_state = (cpos + 1, nr, nc)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append((nr, nc, cpos + 1))
        return False

    X = 0
    n = 1
    while n < 10000:  
        if not can_form(n):
            return (X,)
        X = n
        n += 1
    return (X,)


toolbox.register("evaluate", eval_814_num)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=9, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    loaded = []
    if os.path.exists('best_output.txt'):
        with open('best_output.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ind_list = list(map(int, line.split(',')))
                    if len(ind_list) == IND_SIZE:
                        ind = creator.Individual(ind_list)
                        loaded.append(ind)
                except ValueError:
                    pass  # Skip invalid lines

    num_loaded = len(loaded)
    if num_loaded >= 300:
        pop = random.sample(loaded, 300)
    else:
        pop = loaded[:]
        remaining = 300 - len(pop)
        if remaining > 0:
            pop.extend(toolbox.population(n=remaining))

    fitnesses = list(futures.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    CXPB, MUTPB = 0.5, 0.2
    fits = [ind.fitness.values[0] for ind in pop]
    g = 0
    while max(fits) < 5000 and g < 300:
        g = g + 1
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
        fitnesses = list(futures.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        if g % 10 == 0:
            sys.stdout.write("-- Generation %i --\n" % g)
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            sys.stdout.write(" Min %s\n" % min(fits))
            sys.stdout.write(" Max %s\n" % max(fits))
            sys.stdout.write(" Avg %s\n" % mean)
            sys.stdout.write(" Std %s\n" % std)
    best_ind = tools.selBest(pop, 1)[0]
    sys.stdout.write("Best individual is %s %s\n" % (best_ind, best_ind.fitness.values))
    for i in range(8):
        row = ''.join(map(str, best_ind[i * 14:(i + 1) * 14]))
        sys.stdout.write(row + '\n')

    with open('best_output.txt', 'a') as f:
        f.write(','.join(map(str, best_ind)) + '\n')


if __name__ == "__main__":
    main()