import random
import numpy as np
import sys
import os
import copy
from collections import Counter
from deap import base, creator, tools
import torch
import torch.nn.functional as F

# --- Device & Grid setup ---
DEVICE = 'cuda'
IND_ROWS, IND_COLS = 8, 14
IND_SIZE = IND_ROWS * IND_COLS
TARGET_POP = 500
FILENAME = 'best_output_double.txt'
MAX_N = 50000

KERNEL = torch.ones((1, 1, 3, 3), device=DEVICE)
KERNEL[0, 0, 1, 1] = 0

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def load_previous_best():
    loaded = []
    if os.path.exists(FILENAME):
        with open(FILENAME, 'r') as f:
            valid_lines = [line.strip() for line in f if len(line.strip()) == 14 and line.strip().isdigit()]
        for block_start in range(0, len(valid_lines), 8):
            if block_start + 7 >= len(valid_lines): break
            block = valid_lines[block_start:block_start + 8]
            ind_list = [int(d) for row in block for d in row]
            if len(ind_list) == IND_SIZE:
                loaded.append(creator.Individual(ind_list))
    return loaded

def check_paths_multi_batch(grids, num_strs):
    B = grids.shape[0]
    num_count = len(num_strs)
    results = torch.zeros((B, num_count), dtype=torch.bool, device=DEVICE)
    for i, n_str in enumerate(num_strs):
        digits = [int(d) for d in n_str]
        mask = (grids == digits[0]).float()
        for d in digits[1:]:
            if not mask.any():
                mask = None
                break
            spread = F.conv2d(mask, KERNEL, padding=1) > 0
            mask = (spread & (grids == d)).float()
        if mask is not None:
            results[:, i] = mask.view(B, -1).sum(dim=1) > 0
    return results

def evaluate_batch_gpu(ind_lists):
    if not ind_lists: return []
    grids = torch.tensor(ind_lists, dtype=torch.int8, device=DEVICE).view(-1, 1, IND_ROWS, IND_COLS)
    B = grids.shape[0]
    current_scores = torch.zeros(B, device=DEVICE)
    still_alive = torch.ones(B, dtype=torch.bool, device=DEVICE)
    for n in range(1, MAX_N + 1):
        if not still_alive.any(): break
        exists = check_paths_multi_batch(grids, [str(n)])[:, 0]
        current_scores = torch.where(exists & still_alive, current_scores + 1, current_scores)
        still_alive &= exists
    return current_scores.cpu().numpy()

def update_crowding(population):
    fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
    for front in fronts:
        tools.emo.assignCrowdingDist(front)
    return [ind for front in fronts for ind in front]

def custom_mate(ind1, ind2):
    grid1, grid2 = np.array(ind1).reshape(IND_ROWS, IND_COLS), np.array(ind2).reshape(IND_ROWS, IND_COLS)
    sy, sx = random.randint(0, IND_ROWS-2), random.randint(0, IND_COLS-2)
    ey, ex = random.randint(sy+1, IND_ROWS), random.randint(sx+1, IND_COLS)
    grid1[sy:ey, sx:ex], grid2[sy:ey, sx:ex] = grid2[sy:ey, sx:ex].copy(), grid1[sy:ey, sx:ex].copy()
    ind1[:], ind2[:] = grid1.flatten().tolist(), grid2.flatten().tolist()
    return ind1, ind2

def custom_mutate(individual):
    if random.random() < 0.5: tools.mutUniformInt(individual, 0, 9, 0.05)
    else:
        a, b = random.sample(range(10), 2)
        for i in range(len(individual)):
            if individual[i] == a: individual[i] = b
            elif individual[i] == b: individual[i] = a
    return individual,

toolbox.register("attr_int", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournamentDCD)

def main():
    loaded_inds = load_previous_best()
    pop = loaded_inds + toolbox.population(n=max(0, TARGET_POP - len(loaded_inds)))
    pop = pop[:TARGET_POP]
    
    # 초기 평가 및 max_score_all_time 설정
    scores = evaluate_batch_gpu([list(ind) for ind in pop])
    for ind, s in zip(pop, scores): ind.fitness.values = (s,)
    
    max_score_all_time = max([ind.fitness.values[0] for ind in pop]) if pop else -1.0
    print(f"Initial Max Score: {max_score_all_time}")
    
    pop = update_crowding(pop)
    best_ind = None
    improved = False

    for g in range(1, 30001):
        offspring = [copy.deepcopy(ind) for ind in toolbox.select(pop, len(pop))]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offspring:
            if random.random() < 0.2:
                toolbox.mutate(m)
                del m.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            i_scores = evaluate_batch_gpu([list(ind) for ind in invalid_ind])
            for ind, s in zip(invalid_ind, i_scores): ind.fitness.values = (s,)

        pop = update_crowding(pop + offspring)[:TARGET_POP]
        current_max = max([ind.fitness.values[0] for ind in pop])
        
        if current_max > max_score_all_time:
            print(f"★ NEW RECORD! Gen {g}: {max_score_all_time} -> {current_max}")
            max_score_all_time = current_max
            best_ind = copy.deepcopy(pop[np.argmax([ind.fitness.values[0] for ind in pop])])
            improved = True

        if g % 500 == 0:
            print(f"Gen {g:3d} | Max: {max_score_all_time}")

    if not improved:
        print("\n" + "!"*60)
        print("""
   _______  _______  ___   ___      
  |       ||   _   ||   | |   |     
  |    ___||  |_|  ||   | |   |     
  |   |___ |       ||   | |   |     
  |    ___||       ||   | |   |___  
  |   |    |   _   ||   | |       | 
  |___|    |__| |__||___| |_______| 

  NO PROGRESS IN THIS RUN!
        """)
        print("!"*60 + "\n")

    if best_ind and improved:
        grid = np.array(best_ind).reshape(IND_ROWS, IND_COLS)
        with open(FILENAME, 'a') as f:
            for row in grid: 
                f.write(''.join(map(str, row)) + '\n')
                print(''.join(map(str, row)))
            f.write(f"\n")
        print(f"Final Success: {max_score_all_time} saved.")

if __name__ == "__main__":
    main()
