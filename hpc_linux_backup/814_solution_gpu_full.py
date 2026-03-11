import random
import numpy as np
import sys
import os
import copy
from collections import Counter
from deap import base, creator, tools
import torch
import torch.nn.functional as F

# --- Device setup (CUDA is guaranteed) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using PyTorch device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# --- Grid constants ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
TARGET_POP = 500
FILENAME = 'best_output_double.txt'
MAX_N = 50000
NGEN = 100
G_PRINT_GROUP = 1
G_LOCAL_THRESHOLD = 500
STAGNATION_LIMIT = 200
G_BOOM_THRESHOLD = 2000

# Convolution kernel for 8-direction spread
KERNEL = torch.ones((1, 1, 3, 3), device=DEVICE)
KERNEL[0, 0, 1, 1] = 0

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


# --- Convolution-based path check ---
def check_path_parallel(grids, num_str):
    if not num_str:
        return torch.zeros(grids.shape[0], dtype=torch.bool, device=DEVICE)

    digits = [int(d) for d in num_str]

    mask = (grids == digits[0]).float()  # (B, 1, H, W)

    for next_digit in digits[1:]:
        spread = F.conv2d(mask, KERNEL, padding=1) > 0
        mask = (spread & (grids == next_digit)).float()

        if not mask.any():
            return torch.zeros(grids.shape[0], dtype=torch.bool, device=DEVICE)

    return mask.view(grids.shape[0], -1).sum(dim=1) > 0

def evaluate_batch_gpu(ind_lists):
    if not ind_lists:
        return np.array([]), np.array([])
    grids = torch.tensor(ind_lists, dtype=torch.long, device=DEVICE).view(-1, 1, IND_ROWS, IND_COLS)
    B = grids.shape[0]
    # --------------------------
    # 1) Consecutive score (forward only)
    # --------------------------
    current_scores = torch.zeros(B, device=DEVICE)
    still_alive = torch.ones(B, dtype=torch.bool, device=DEVICE)
    n = 1
    while n <= MAX_N and still_alive.any():
        exists = check_path_parallel(grids, str(n))  # (B,) bool
        current_scores[exists & still_alive] += 1
        still_alive &= exists
        n += 1
    # --------------------------
    # 2) Formable count (1000~9999, forward only)
    # --------------------------
    formable_counts = torch.zeros(B, device=DEVICE)
    if n >= 1000:
        formable_counts += n - 999
    for num in range(n + 1, 10000):
        exists = check_path_parallel(grids, str(num))
        formable_counts += exists.float()
    return current_scores.cpu().numpy(), formable_counts.cpu().numpy()


# --- Custom selection with stagnation handling ---
def custom_select(pop, stagnation_counter):
    target_size = len(pop)
    if stagnation_counter >= STAGNATION_LIMIT:
        r = random.random()
        if r < 0.1:
            return tools.selTournamentDCD(pop, target_size)
        elif r < 0.4:
            return tools.selWorst(pop, target_size)
        elif r < 0.7:
            return tools.selNSGA2(pop, target_size)
        else:
            return tools.selRandom(pop, target_size)
    else:
        return tools.selTournamentDCD(pop, target_size)


def update_crowding(population):
    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    assign = tools.emo.assignCrowdingDist
    for front in fronts:
        assign(front)
    return [ind for front in fronts for ind in front]


# --- Load previous best ---
def load_previous_best():
    loaded = []
    if os.path.exists(FILENAME):
        with open(FILENAME, 'r') as f:
            valid_lines = [line.strip() for line in f if len(line.strip()) == 14 and line.strip().isdigit()]
        for block_start in range(0, len(valid_lines), 8):
            if block_start + 7 >= len(valid_lines):
                break
            block = valid_lines[block_start:block_start + 8]
            ind_list = []
            for row in block:
                ind_list.extend(int(d) for d in row)
            if len(ind_list) == IND_SIZE:
                ind = creator.Individual(ind_list)
                loaded.append(ind)
    return loaded


# --- GA operators ---
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


toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournamentDCD)
toolbox.register("attr_int", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# --- Main ---
def main():
    loaded_inds = load_previous_best()
    pop = loaded_inds + toolbox.population(n=max(0, TARGET_POP - len(loaded_inds)))
    pop = pop[:TARGET_POP]

    ind_lists = [list(ind) for ind in pop]
    scores, forms = evaluate_batch_gpu(ind_lists)
    for ind, s, f in zip(pop, scores, forms):
        ind.fitness.values = (s, f)

    update_crowding(pop)

    max_score_all_time = max([ind.fitness.values[0] for ind in pop]) if pop else -1.0
    sys.stdout.write(f"Initial Max Score: {max_score_all_time:.0f}\n")
    sys.stdout.flush()

    best_ind = None
    local_max_ind = None
    local_max_score = -1.0
    improved = False
    stagnation_counter = 0
    last_max = max_score_all_time

    for g in range(1, NGEN + 1):
        offspring = custom_select(pop, stagnation_counter)
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            ind_lists = [list(ind) for ind in invalid_ind]
            scores, forms = evaluate_batch_gpu(ind_lists)
            for ind, s, f in zip(invalid_ind, scores, forms):
                ind.fitness.values = (s, f)

        pop[:] = offspring
        pop = update_crowding(pop)

        current_max = max([ind.fitness.values[0] for ind in pop])

        if current_max > max_score_all_time:
            print(f"★ NEW RECORD! Gen {g}: {max_score_all_time} -> {current_max}")
            local_max_score = current_max
            max_score_all_time = current_max
            best_ind = toolbox.clone(pop[np.argmax([ind.fitness.values[0] for ind in pop])])
            improved = True
            stagnation_counter = 0
            last_max = current_max
        else:
            if current_max <= last_max:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_max = current_max

        if g > G_LOCAL_THRESHOLD and current_max >= max_score_all_time * 0.8:
            if current_max > local_max_score:
                local_max_score = current_max
                local_max_ind = toolbox.clone(pop[np.argmax([ind.fitness.values[0] for ind in pop])])

        if g % G_PRINT_GROUP == 0:
            sys.stdout.write(f"Gen {g:5d} | Current Max: {current_max:.0f} "
                             f"| All-time Max: {max_score_all_time:.0f} "
                             f"| local max Score : {local_max_score:.0f}\n")
      
            sys.stdout.flush()
            
        if g > G_BOOM_THRESHOLD and local_max_score < 0:
            sys.stdout.write("\n" + "!" * 60 + "\n")
            sys.stdout.write(f"BOOM wtfffffffff BOOM")
            sys.stdout.write("\n" + "!" * 60 + "\n")
            sys.stdout.flush()
            break;
          
          
    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write(f"Evolution finished after {NGEN} generations\n")
    sys.stdout.flush()

    saved_ind = best_ind if (best_ind and improved) else local_max_ind
    if saved_ind:
        grid = np.array(saved_ind).reshape(IND_ROWS, IND_COLS)
        with open(FILENAME, 'a') as f:
            for row in grid:
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
        print(f"Final Saved (Score: {saved_ind.fitness.values[0]:.0f}, Form: {saved_ind.fitness.values[1]:.0f})")
    else:
        print("\nNO PROGRESS OR NO CANDIDATE TO SAVE!")


if __name__ == "__main__":
    main()