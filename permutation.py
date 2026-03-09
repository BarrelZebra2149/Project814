import numpy as np
import itertools
from numba import njit
from tqdm import tqdm
import os

# =============================================================================
# 1. 파일에서 격자 읽기 로직
# =============================================================================
def load_all_grids(filename):
    """파일 내의 모든 8x14 격자를 추출하여 리스트로 반환"""
    grids = []
    if not os.path.exists(filename):
        print(f"파일이 없습니다: {filename}")
        return grids

    with open(filename, 'r') as f:
        current_grid = []
        for line in f:
            line = line.strip()
            # 14자리 숫자열인지 확인
            if len(line) == 14 and line.isdigit():
                current_grid.append([int(d) for d in line])
                if len(current_grid) == 8:
                    grids.append(np.array(current_grid, dtype=np.int64).flatten())
                    current_grid = []
    return grids


# =============================================================================
# 2. 고속 매핑 및 평가 엔진
# =============================================================================
@njit(fastmath=True)
def _has_path_fast(grid, digits):
    """Ultra-fast pathfinding using static stack."""
    rows, cols = grid.shape
    digit_len = digits.shape[0]
    deltas = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int64)
    stack = np.empty((500, 3), dtype=np.int64)
    head = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == digits[0]:
                if digit_len == 1: return True
                stack[head, 0], stack[head, 1], stack[head, 2] = r, c, 0
                head += 1
    while head > 0:
        head -= 1
        r, c, idx = stack[head, 0], stack[head, 1], stack[head, 2]
        next_digit = digits[idx + 1]
        for i in range(8):
            nr = r + deltas[i, 0]
            nc = c + deltas[i, 1]
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == next_digit:
                if idx + 1 == digit_len - 1: return True
                stack[head, 0], stack[head, 1], stack[head, 2] = nr, nc, idx + 1
                head += 1
    return False

@njit(fastmath=True)
def _get_digits_math(n):
    temp = np.empty(6, dtype=np.int64)
    idx = 0
    while n > 0:
        temp[idx] = n % 10
        n //= 10
        idx += 1
    result = np.empty(idx, dtype=np.int64)
    for i in range(idx):
        result[i] = temp[idx - 1 - i]
    return result


@njit(fastmath=True)
def _reverse_int_math(n):
    rev = 0
    while n > 0:
        rev = rev * 10 + (n % 10)
        n //= 10
    return rev


@njit
def _evaluate_core_numba(grid_1d, rows, cols):
    grid = grid_1d.reshape((rows, cols))
    found = np.zeros(50000, dtype=np.bool_)
    current_score, n = 49999, 1
    while n < 50000:
        rev_n = _reverse_int_math(n)
        if found[n] or (n % 10 != 0 and rev_n < 50000 and found[rev_n]):
            n += 1
            continue
        digits = _get_digits_math(n)
        if _has_path_fast(grid, digits):
            found[n] = True
            if rev_n < 50000: found[rev_n] = True
        else:
            current_score = n - 1
            break
        n += 1
    formable = max(0, min(30000, current_score) - 1000 + 1)
    for num in range(max(1000, current_score + 1), 30000):
        rev_num = _reverse_int_math(num)
        if found[num] or (num % 10 != 0 and rev_num < 50000 and found[rev_num]):
            formable += 1
            continue
        digits = _get_digits_math(num)
        if _has_path_fast(grid, digits):
            formable += 1
            found[num] = True
            if rev_num < 50000: found[rev_num] = True
    return float(current_score), float(formable)

@njit(fastmath=True)
def evaluate_mapping(grid_1d, p_array):
    """주어진 순열 p_array를 적용하여 즉시 평가"""
    # 벡터화된 매핑 적용
    mapped = np.empty_like(grid_1d)
    for i in range(len(grid_1d)):
        mapped[i] = p_array[grid_1d[i]]

    # 기존 유저님의 평가 함수 호출 (이 스코프 내에 있어야 함)
    score, formable = _evaluate_core_numba(mapped, 8, 14)
    return score, formable, mapped


# =============================================================================
# 3. 메인 프로세서
# =============================================================================
def run_brute_force_optimization(input_file, output_file):
    grids = load_all_grids(input_file)
    print(f"총 {len(grids)}개의 격자를 발견했습니다. 전수 조사를 시작합니다.")

    digits = np.arange(10, dtype=np.int64)
    all_perms = list(itertools.permutations(digits))  # 3,628,800개

    with open(output_file, 'w') as f_out:
        for idx, grid in enumerate(grids):
            print(f"\n[격자 {idx + 1}/{len(grids)}] 최적 맵핑 찾는 중...")

            best_score = -1.0
            best_mapped_grid = None
            best_mapping = None

            # 10! 루프
            for p in tqdm(all_perms, desc="Permutations", leave=False):
                p_array = np.array(p, dtype=np.int64)
                score, formable, mapped = evaluate_mapping(grid, p_array)

                if score > best_score:
                    best_score = score
                    best_mapped_grid = mapped
                    best_mapping = p_array

            # 결과 저장 및 출력
            print(f"결과: {best_score:.0f} (Mapping: {best_mapping})")

            f_out.write(f"--- Grid {idx + 1} Optimized (Score: {best_score:.0f}) ---\n")
            grid_2d = best_mapped_grid.reshape(8, 14)
            for row in grid_2d:
                f_out.write("".join(map(str, row)) + "\n")
            f_out.write(f"Mapping Used: {best_mapping}\n\n")
            f_out.flush()


if __name__ == "__main__":
    # 실행 전 _evaluate_core_numba 함수들이 정의되어 있어야 합니다.
    run_brute_force_optimization('test.txt', 'optimized_results.txt')