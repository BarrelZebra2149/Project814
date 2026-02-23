import random
import numpy as np
import sys
import os
import multiprocessing
from collections import Counter
from deap import base, creator, tools
import torch
import torch.nn.functional as F

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # already set

# --- Device setup (CUDA is guaranteed) ---
DEVICE = 'cpu'
print(f"Using device: {DEVICE}")
#print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Grid constants ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
INT_MIN, INT_MAX = 0, 9
FILENAME = 'best_output_double.txt'

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Convolution kernel for 8-direction spread
KERNEL = torch.ones((1, 1, 3, 3), device=DEVICE)
KERNEL[0, 0, 1, 1] = 0

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


# ====================== GLOBAL PRECOMPUTATION (한 번만 실행) ======================
REPRESENTATIVES = []
PAIR_DICT = {}  # rep → [forward, reverse] (중복 제거)
seen = set()

for n in range(1, 10001):
    if n in seen:
        continue

    rev_str = str(n)[::-1]
    rev_n = int(rev_str) if not rev_str.startswith('0') else None

    covered = [n]
    if rev_n is not None and rev_n != n:
        covered.append(rev_n)
        seen.add(rev_n)

    seen.add(n)
    REPRESENTATIVES.append(n)
    PAIR_DICT[n] = covered

print(f"Precomputed {len(REPRESENTATIVES)} representatives (reverse dedup)")


# ====================== GPU BATCH EVALUATION (최적화 버전) ======================
def evaluate_batch_gpu(ind_lists):
    if not ind_lists:
        return np.array([]), np.array([])

    grids = torch.tensor(ind_lists, dtype=torch.long, device=DEVICE).unsqueeze(1)  # (B, 1, H, W)
    B = grids.shape[0]

    # 한 번의 대형 배치로 모든 대표자 체크
    has_array = torch.zeros((B, len(REPRESENTATIVES)), dtype=torch.bool, device=DEVICE)

    for i, rep in enumerate(REPRESENTATIVES):
        has_array[:, i] = check_path_parallel(grids, str(rep))

    # 각 그리드별 found_set 구축 (이미 발견된 숫자는 영원히 스킵)
    current_scores = np.zeros(B, dtype=np.float32)
    formable_counts = np.zeros(B, dtype=np.float32)

    for b in range(B):
        found_set = set()

        for j, rep in enumerate(REPRESENTATIVES):
            if has_array[b, j]:
                found_set.update(PAIR_DICT[rep])

        # Consecutive score
        score = 0
        for k in range(1, MAX_N + 1):
            if k in found_set:
                score = k
            else:
                break
        current_scores[b] = score

        # Formable count (1000~9999)
        formable_counts[b] = sum(1 for num in range(1000, 10000) if num in found_set)

    return formable_counts, current_scores
# Register the accelerated evaluate
toolbox.register("evaluate", evaluate_batch_gpu)

# ────────────────────────────────────────────────
# The rest of your original code (unchanged)
# ────────────────────────────────────────────────
def custom_mate(ind1, ind2):
    grid1 = np.array(ind1).reshape(IND_ROWS, IND_COLS)
    grid2 = np.array(ind2).reshape(IND_ROWS, IND_COLS)
    s_y = random.randint(0, IND_ROWS - 1)
    e_y = random.randint(s_y + 1, IND_ROWS)
    s_x = random.randint(0, IND_COLS - 1)
    e_x = random.randint(s_x + 1, IND_COLS)
    temp = grid1[s_y:e_y, s_x:e_x].copy()
    grid1[s_y:e_y, s_x:e_x] = grid2[s_y:e_y, s_x:e_x]
    grid2[s_y:e_y, s_x:e_x] = temp
    ind1[:] = grid1.ravel().tolist()
    ind2[:] = grid2.ravel().tolist()
    return ind1, ind2

def custom_mutate(individual, indpb=0.05):
    if random.random() < 0.5:
        tools.mutUniformInt(individual, low=0, up=9, indpb=indpb)
    else:
        digits = list(range(10))
        random.shuffle(digits)
        a, b = digits[0], digits[1]
        for i in range(len(individual)):
            if individual[i] == a:
                individual[i] = b
            elif individual[i] == b:
                individual[i] = a
    return individual,

toolbox.register("mate", custom_mate)
toolbox.register("mutate", custom_mutate, indpb=0.05)

def _assign_crowding_dist_fallback(front):
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
    fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
    assign = None
    if hasattr(tools, "emo") and hasattr(tools.emo, "assignCrowdingDist"):
        assign = tools.emo.assignCrowdingDist
    for front in fronts:
        if assign is not None:
            assign(front)
        else:
            _assign_crowding_dist_fallback(front)

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

    # Serial evaluation
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    update_crowding(pop)

    CXPB, MUTPB = 0.5, 0.2
    NGEN = 100
    best_current_score = 0.0
    best_ind = None
    for g in range(1, NGEN + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        update_crowding(pop)
        pop_max_v = [ind.fitness.values[0] for ind in pop]
        current_max = max(pop_max_v)
        if current_max > best_current_score:
            best_current_score = current_max
            best_ind = max(pop, key=lambda ind: ind.fitness.values[0])
        if g % 1 == 0:
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
        with open(FILENAME, 'a') as f:
            for row in grid:
                f.write(''.join(map(str, row)) + '\n')
            f.write('\n')
        print("Highest current_score grid saved to file.")
    else:
        print("No valid individuals found to save.")

    top3 = tools.selBest(pop, k=3)
    for rank, ind in enumerate(top3, 1):
        curr, form = ind.fitness.values
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

    with open(FILENAME, 'a') as f:
        grid = np.array(top3[0]).reshape(IND_ROWS, IND_COLS)
        for row in grid:
            f.write(''.join(map(str, row)) + '\n')
        f.write('\n')

if __name__ == "__main__":
    main()