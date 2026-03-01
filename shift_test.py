import numpy as np
import random
IND_ROWS = 8
IND_COLS = 14

# ====================== 테스트용 함수 복사 ======================
def directional_cyclic_one_shift_mutation(grid):
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    r = random.randint(0, IND_ROWS - 1)
    c = random.randint(0, IND_COLS - 1)
    dr, dc = random.choice(deltas)

    def get_line(r, c, dr, dc):
        line = []

        cr, cc = r - dr, c - dc
        while 0 <= cr < IND_ROWS and 0 <= cc < IND_COLS:
            line.insert(0, (cr, cc))
            cr -= dr
            cc -= dc
        line.append((r, c))

        cr, cc = r + dr, c + dc
        while 0 <= cr < IND_ROWS and 0 <= cc < IND_COLS:
            line.append((cr, cc))
            cr += dr
            cc += dc
        return line

    line = get_line(r, c, dr, dc)

    if len(line) < 3:
        r = random.randint(1, IND_ROWS - 2)
        c = random.randint(1, IND_COLS - 2)
        dr, dc = random.choice(deltas)
        line = get_line(r, c, dr, dc)

    values = [grid[pr, pc] for pr, pc in line]
    shifted = [values[-1]] + values[:-1]

    for i, (pr, pc) in enumerate(line):
        grid[pr, pc] = shifted[i]

    # 테스트 정보 출력
    print(f"\n=== SHIFT 발생 ===")
    print(f"시작 위치: ({r}, {c})")
    print(f"방향 (dr,dc): ({dr}, {dc})")
    print(f"라인 길이: {len(line)}")
    print(f"원래 값:  {values}")
    print(f"Shift 후: {shifted}")
    print(f"Shifted 라인 위치: {line}")
    print("==================\n")


# ====================== 테스트 실행 ======================
def print_grid(grid, title=""):
    print(f"\n{title}")
    for row in grid:
        print(' '.join(f"{int(x):1d}" for x in row))
    print("-" * 60)


if __name__ == "__main__":
    random.seed(42)  # 재현 가능하게

    # 8x14 랜덤 그리드 생성 (0~9)
    grid = np.random.randint(0, 10, size=(8, 14), dtype=int)

    print_grid(grid, "=== BEFORE MUTATION ===")

    # 10번 연속 테스트 (여러 라인 확인용)
    for i in range(10):
        print(f"\n=== 테스트 {i + 1}/10 ===")
        directional_cyclic_one_shift_mutation(grid)
        print_grid(grid, f"AFTER TEST {i + 1}")