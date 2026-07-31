# Project814 — BOJ 18789 (814-2)

8×14 격자에 0~9 숫자를 배치해, 1부터 연속으로 큰 수까지 8방향 인접 워크로 읽어낼 수
있게 만드는 최적화 문제 (문제 조건: [블로그 풀이](https://velog.io/@spark1130/%EB%B0%B1%EC%A4%80-814-2-%EB%AC%B8%EC%A0%9C-%ED%92%80%EC%9D%B4)).
점수 = 1..K가 전부 형성 가능한 최대 K (이론 최대 8142).

## 디렉터리 구조

```
school_linux_backup/
├── code/       원본 시도: DEAP 기반 GA (814_cpu_score_first.py / 814_cpu_count_first.py),
│               DLAS C++ 구현(dlas.hpp, 814_clang.cpp, my_dlas 바이너리), 보조 스크립트
│               (check_grids.py, make_new_gen.py, permutation.py)
├── data/       격자 코퍼스 (*.txt, 8줄×14자리 블록 형식)
├── sa814/      새로 작성한 SA/병렬 템퍼링 솔버 — 지금은 이쪽을 쓰세요
└── requirements.txt   code/ 아래 원본 GA용 (numpy, deap, numba, tqdm)
```

`code/`의 원본 GA는 온도/담금질 개념이 없고, 실제 점수 계산은 Linux 전용 `my_dlas`
바이너리에 subprocess로 위임했었습니다 (Windows에서는 애초에 동작 안 함). `sa814/`는
이를 대체하는 SA(담금질) + 병렬 템퍼링 엔진으로, 실제 점수를 파이썬/numba 안에서 직접
계산하고 체크포인트/재개도 지원합니다. `code/`는 비교 기준선으로 그대로 남아 있습니다.

## 빠른 시작 (sa814)

**Windows:**
```bash
cd sa814
pip install numpy numba tqdm
python win_score_first.py --seconds 3600
```

**Linux (venv + tmux):**
```bash
cd sa814
python3 -m venv .venv
source .venv/bin/activate
pip install numpy numba tqdm

tmux new -s sa814-score
# tmux 세션 안에서:
source .venv/bin/activate
python3 linux_score_first.py --run-name score_run1
# 빠져나오기: Ctrl+b d
# 다시 들어가기: tmux attach -t sa814-score
```

같은 명령을 다시 실행하면 `--resume`(기본값)으로 마지막 체크포인트에서 이어서
진행됩니다. 결과는 `sa814/runs/<run_name>/best.txt`(제출용 격자),
`records.txt`(신기록 이력), `progress.csv`(진행 로그)에 쌓입니다.

자세한 설계 근거, 검증 결과, 전체 CLI 옵션은 [sa814/README.md](sa814/README.md) 참고.

## 검증

```bash
cd sa814
python verify_scorer.py   # 새 채점기 == 원본 채점기, 4개 계층 교차검증
python bench.py           # 채점기/SA 처리량 벤치마크
```
