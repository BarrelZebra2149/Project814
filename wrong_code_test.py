import numpy as np
from deap import base, creator, tools
# --- Constants ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
FILENAME = 'test.txt'

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

import random
import numpy as np
import time
from collections import Counter
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- Constants ---
IND_ROWS = 8
IND_COLS = 14
IND_SIZE = IND_ROWS * IND_COLS
MAX_N = 10000

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
FILENAME = 'test.txt'

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
# Match CPU: when n is formable, add both n and rev(n) to found_set.
# rev(n) is int(str(n)[::-1]) so e.g. rev(10)=1, rev(100)=1.
REPRESENTATIVES = list(range(1, 10001))
PAIR_DICT = {}
for n in REPRESENTATIVES:
    rev_n = int(str(n)[::-1])  # 10 -> 1, 100 -> 1, etc.
    covered = [n]
    if rev_n != n:
        covered.append(rev_n)
    PAIR_DICT[n] = covered

print(f"Precomputed {len(REPRESENTATIVES)} representatives (1..10000, reverse pairs in PAIR_DICT)")

import torch


# --- 초기 설정 (프로그램 시작 시 한 번만 실행) ---
def prepare_mapping_matrix(REPRESENTATIVES, PAIR_DICT, MAX_NUM=10000):
    # (대표자 수, 전체 숫자 범위) 크기의 행렬
    mapping = torch.zeros((len(REPRESENTATIVES), MAX_NUM + 1), device=DEVICE)

    for i, rep in enumerate(REPRESENTATIVES):
        if rep in PAIR_DICT:
            for num in PAIR_DICT[rep]:
                if num <= MAX_NUM:
                    mapping[i, num] = 1.0
    return mapping


# 전역 변수 등으로 미리 생성해둡니다.
MAPPING_MATRIX = prepare_mapping_matrix(REPRESENTATIVES, PAIR_DICT)


def evaluate_batch_gpu(ind_lists):
    if not ind_lists:
        return np.array([]), np.array([])

    # 1. 그리드 생성 및 대표자 체크 (기존과 동일)
    grids = torch.tensor(ind_lists, dtype=torch.long, device=DEVICE).view(-1, 1, IND_ROWS, IND_COLS)
    B = grids.shape[0]

    has_array = torch.zeros((B, len(REPRESENTATIVES)), dtype=torch.float32, device=DEVICE)
    for i, rep in enumerate(REPRESENTATIVES):
        has_array[:, i] = check_path_parallel(grids, str(rep)).float()

    found_matrix = (torch.matmul(has_array, MAPPING_MATRIX) > 0)
    consecutive_part = found_matrix[:, 1:MAX_N + 1].long()
    consecutive_mask = torch.cumprod(consecutive_part, dim=1)
    current_scores = torch.sum(consecutive_mask, dim=1)
    formable_counts = torch.sum(found_matrix[:, 1000:10000], dim=1)

    return current_scores.cpu().numpy(), formable_counts.cpu().numpy()

toolbox.register("evaluate", evaluate_batch_gpu)


def benchmark_eval(individual, repeats=1):
    times = []
    for _ in range(repeats):
        start = time.time()
        current_scores , formable_counts = evaluate_batch_gpu(individual)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / repeats
    return avg_time, current_scores, formable_counts

# --- Run Tests ---
if __name__ == "__main__":

    print("=== Evaluation Function Benchmark ===\n")
    avg_time, current_score, formable_count = benchmark_eval(load_previous_best(), repeats=1)

    print(f"  Current Score : {current_score[0]:.0f}")
    print(f"  Formable Count: {formable_count[0]:.0f}")
    print(f"  Eval Time : {avg_time:.4f} seconds")
    print("-" * 50)

    print("Benchmark complete. Check times for bottlenecks.")
    print("If 'worst' case is very slow, reverse opt helps less there (few discoveries).")