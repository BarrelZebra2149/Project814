import numpy as np
import sys
from collections import deque


def load_grid(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if len(line.strip()) == 14]
    if len(lines) < 8:
        raise ValueError("파일은 정확히 8줄, 각 줄 14자리 숫자여야 합니다.")

    grid = np.zeros((8, 14), dtype=int)
    for i in range(8):
        for j in range(14):
            grid[i, j] = int(lines[i][j])
    return grid


def has_path_and_positions(grid: np.ndarray, digits: list[int]):
    """경로가 있으면 (True, 사용된 위치 리스트) 반환, 없으면 (False, None)"""
    if not digits:
        return False, None

    rows, cols = grid.shape
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for sr in range(rows):
        for sc in range(cols):
            if grid[sr, sc] != digits[0]:
                continue

            queue = deque([(sr, sc, 0, [(sr, sc)])])
            visited = set([(sr, sc, 0)])

            while queue:
                r, c, idx, path = queue.popleft()
                if idx == len(digits) - 1:
                    return True, path

                next_d = digits[idx + 1]
                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            grid[nr, nc] == next_d and (nr, nc, idx + 1) not in visited):
                        visited.add((nr, nc, idx + 1))
                        queue.append((nr, nc, idx + 1, path + [(nr, nc)]))
    return False, None


def brute_force_position(grid: np.ndarray, r: int, c: int):
    """특정 위치 (r, c)를 0~9로 바꿔보며 연속 점수와 성공 개수 변화 확인"""
    original_val = grid[r, c]
    original_max_consecutive = calculate_max_consecutive(grid)
    original_success = calculate_success_count(grid)

    print(f"\n=== 브루트포스 분석: 위치 ({r}, {c}) ===")
    print(f"원본 값: {original_val}")
    print(f"원본 연속 점수: {original_max_consecutive:6,d} 점")
    print(f"원본 1,000~99,999 성공 개수: {original_success:6,d} 개")
    print(f"{'-' * 50}")

    for new_val in range(10):
        if new_val == original_val:
            continue  # 원본은 스킵

        grid[r, c] = new_val
        new_max_consecutive = calculate_max_consecutive(grid)
        new_success = calculate_success_count(grid)

        consecutive_delta = new_max_consecutive - original_max_consecutive
        success_delta = new_success - original_success

        print(f"새 값 {new_val}: 연속 점수 {new_max_consecutive:6,d} ({consecutive_delta:+6,d})")
        print(f"           성공 개수 {new_success:6,d} ({success_delta:+6,d})\n")

    # 원본 복원
    grid[r, c] = original_val


def calculate_max_consecutive(grid: np.ndarray) -> int:
    """1부터 연속으로 성공한 최대 점수 계산"""
    max_consecutive = 0
    n = 1
    while n < 50000:
        digits = [int(d) for d in str(n)]
        found, _ = has_path_and_positions(grid, digits)
        if found:
            max_consecutive = n
            n += 1
        else:
            break
    return max_consecutive


def calculate_success_count(grid: np.ndarray) -> int:
    """1,000~99,999 성공 개수 계산"""
    success = 0
    for n in range(1000, 10000):
        digits = [int(d) for d in str(n)]
        found, _ = has_path_and_positions(grid, digits)
        if found:
            success += 1
    return success


def main():
    grid = load_grid('test.txt')

    position_count = np.zeros((8, 14), dtype=int)
    thousand_count = [0] * 101
    success_1000_99999 = 0

    print("🔍 100,000점 고지를 향한 정밀 분석 시작 (약간의 시간이 소요될 수 있습니다)...\n")

    # 1. 실제 연속 점수 계산 (1부터 끊길 때까지)
    max_consecutive = calculate_max_consecutive(grid)

    # 2. 전체 범위(1000 ~ 99999) 검사 + 기여도 계산
    for n in range(1000, 100000):
        digits = [int(d) for d in str(n)]
        found, positions = has_path_and_positions(grid, digits)
        if found:
            success_1000_99999 += 1
            thousand_count[n // 1000] += 1
            for r, c in positions:
                position_count[r, c] += 1

    # ====================== 출력부 ======================
    print(f"{'=' * 95}")
    print(f"🏆 격자 최종 연속 점수      : {max_consecutive:6,d} 점")
    print(f"📈 1,000~99,999 구간 성공   : {success_1000_99999:6,d} 개 / 99,000 개 "
          f"({success_1000_99999 / 99000 * 100:6.2f}%)")
    print(f"{'=' * 95}\n")

    print("📊 [만 단위별 요약]")
    for i in range(0, 10):
        start = i * 10000 + 1000 if i == 0 else i * 10000
        end = min((i + 1) * 10000 - 1, 99999)
        range_sum = sum(thousand_count[i * 10: (i + 1) * 10])
        print(f" {start:5,d} ~ {end:5,d} 구간 : {range_sum:5,d} 개 성공")

    print(f"\n🔥 [위치별 기여도] (셀당 경로 점유 횟수)")
    print("-" * 110)
    for row in range(8):
        line = " ".join(f"{position_count[row, col]:7d}" for col in range(14))
        print(line)
    print("-" * 110)
    print("TIP: 숫자가 유난히 높은 셀은 현재 유전자 메타의 '핵심 허브'입니다.")

    # ====================== 브루트포스 부분 ======================
    # 예시 위치 (사용자 쿼리에 지정 안 됨, 그래서 행 0, 열 0으로 테스트)
    for i in range(8):
        for j in range(14):
            brute_force_position(grid, i, j)


if __name__ == "__main__":
    main()