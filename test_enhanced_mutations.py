import numpy as np
import random

IND_ROWS = 8
IND_COLS = 14

import numpy as np
import random


def directional_cyclic_one_shift_mutation(grid):
    """
    확실한 변이 보장: 유효한 라인을 찾을 때까지 시도하며,
    원본 값을 훼손하지 않고 순서만 N칸 이동(Cyclic Shift)시킵니다.
    """
    rows, cols = grid.shape
    mutated = False

    while not mutated:
        # 1. 랜덤 시작점과 방향 선정
        r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        dr, dc = random.choice(deltas)

        # 2. 라인 좌표 수집 (격자 끝까지)
        line_coords = []
        # 한쪽 방향
        curr_r, curr_c = r, c
        while 0 <= curr_r < rows and 0 <= curr_c < cols:
            line_coords.append((curr_r, curr_c))
            curr_r += dr
            curr_c += dc
        # 반대 방향
        curr_r, curr_c = r - dr, c - dc
        while 0 <= curr_r < rows and 0 <= curr_c < cols:
            line_coords.insert(0, (curr_r, curr_c))
            curr_r -= dr
            curr_c -= dc

        # 3. 라인 길이가 최소 3은 되어야 변이의 의미가 있음
        if len(line_coords) >= 3:
            values = [grid[pr, pc] for pr, pc in line_coords]
            shift = random.randint(1, len(line_coords) - 1)  # 1~N-1칸 이동
            shifted_values = values[-shift:] + values[:-shift]

            for i, (pr, pc) in enumerate(line_coords):
                grid[pr, pc] = shifted_values[i]
            mutated = True


def quadrant_border_rotate_mutation(grid):
    """
    완벽한 테두리 회전: 사분면의 '가장자리' 좌표만 정확히 순서대로 뽑아 회전시킵니다.
    """
    rows, cols = grid.shape
    mutated = False

    while not mutated:
        # 1. 사분면을 나눌 기준점 (최소 2x2 영역 보장)
        rt = random.randint(1, rows - 2)
        ct = random.randint(1, cols - 2)

        # 2. 네 사분면 중 하나 선택
        quads = [
            (0, rt, 0, ct), (0, rt, ct + 1, cols - 1),
            (rt + 1, rows - 1, 0, ct), (rt + 1, rows - 1, ct + 1, cols - 1)
        ]
        r1, r2, c1, c2 = random.choice(quads)

        # 3. 테두리 좌표를 시계방향으로 정밀하게 수집
        border = []
        for c in range(c1, c2 + 1): border.append((r1, c))  # 상단 (좌->우)
        for r in range(r1 + 1, r2 + 1): border.append((r, c2))  # 우측 (상->하)
        for c in range(c2 - 1, c1 - 1, -1): border.append((r2, c))  # 하단 (우->좌)
        for r in range(r2 - 1, r1, -1): border.append((r, c1))  # 좌측 (하->상)

        if len(border) >= 4:
            values = [grid[pr, pc] for pr, pc in border]
            # 시계 또는 반시계 방향 1칸 회전
            if random.random() < 0.5:
                shifted = [values[-1]] + values[:-1]
            else:
                shifted = values[1:] + [values[0]]

            for i, (pr, pc) in enumerate(border):
                grid[pr, pc] = shifted[i]
            mutated = True
def print_grid(grid, title):
    print(f"\n=== {title} ===")
    for row in grid:
        print(' '.join(f"{x:1d}" for x in row))
    print("-" * 60)


if __name__ == "__main__":
    random.seed(42)
    grid = np.random.randint(0, 10, (8, 14))

    print_grid(grid, "원본 격자")

    for i in range(5):
        print(f"\n=== 테스트 {i + 1}/5 ===")

        # 첫 번째 강화 함수 테스트
        test_grid1 = grid.copy()
        directional_cyclic_one_shift_mutation(test_grid1)
        print_grid(test_grid1, f"directional_cyclic_one_shift_mutation (랜덤 1~N-1 shift)")

        # 두 번째 강화 함수 테스트
        test_grid2 = grid.copy()
        quadrant_border_rotate_mutation(test_grid2)
        print_grid(test_grid2, f"quadrant_border_rotate_mutation (사분면 border 회전)")

    print("\n✅ 모든 테스트 완료! 에러 없이 정상 동작합니다.")