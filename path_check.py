import numpy as np
import sys
from collections import deque

def load_grid(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        # 빈 줄이나 구분선을 제외하고 실제 데이터 줄만 필터링
        lines = [line.strip() for line in f if len(line.strip()) == 14 and line.strip().isdigit()]
    if len(lines) < 8:
        raise ValueError("파일에 유효한 격자 데이터(8줄 x 14자리)가 부족합니다.")

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

def main():
    # 분석할 파일명을 입력하세요
    try:
        grid = load_grid('test.txt')
    except FileNotFoundError:
        print("Error: 'test.txt' 파일을 찾을 수 없습니다.")
        return

    position_count = np.zeros((8, 14), dtype=int)
    # 100,000까지 커버하기 위해 리스트 크기 확장 (0~99 index)
    thousand_count = [0] * 101
    success_total = 0

    print("🔍 100,000점 고지를 향한 정밀 분석 시작 (약간의 시간이 소요될 수 있습니다)...\n")

    # 1. 실제 연속 점수 계산 (1부터 끊길 때까지)
    max_consecutive = 0
    n = 1
    while n < 100000:
        digits = [int(d) for d in str(n)]
        found, _ = has_path_and_positions(grid, digits)
        if found:
            max_consecutive = n
            n += 1
        else:
            break

    # 2. 전체 범위(1000 ~ 99999) 검사 + 기여도 계산
    for n in range(1000, 100000):
        digits = [int(d) for d in str(n)]
        found, positions = has_path_and_positions(grid, digits)
        if found:
            success_total += 1
            thousand_count[n // 1000] += 1
            for r, c in positions:
                position_count[r, c] += 1

    # ====================== 출력부 ======================
    print(f"{'=' * 95}")
    print(f"🏆 격자 최종 연속 점수      : {max_consecutive:6,d} 점")
    print(f"📈 1,000~99,999 구간 성공   : {success_total:6,d} 개 / 99,000 개 "
          f"({success_total / 99000 * 100:6.2f}%)")
    print(f"{'=' * 95}\n")

    print("📊 [만 단위별 요약]")
    for i in range(0, 10):
        start = i * 10000
        end = start + 9999
        # 해당 만 단위 구간의 천 단위 데이터 합산
        range_sum = sum(thousand_count[i*10 : (i+1)*10])
        if i == 0: start = 1000 # 0~999는 제외
        print(f" {start:5,d} ~ {end:5,d} 구간 : {range_sum:5,d} 개 성공")

    print(f"\n🔥 [위치별 기여도] (셀당 경로 점유 횟수)")
    print("-" * 110)
    for row in range(8):
        line = " ".join(f"{position_count[row, col]:7d}" for col in range(14))
        print(line)
    print("-" * 110)
    print("TIP: 숫자가 유난히 높은 셀은 현재 유전자 메타의 '핵심 허브'입니다.")

if __name__ == "__main__":
    main()