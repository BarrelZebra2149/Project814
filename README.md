# Project814

BOJ 18789 (814-2) 최적화 문제를 여러 환경에서 풀어본 기록 저장소입니다. 8×14 격자에
0~9 숫자를 배치해, 1부터 연속으로 큰 수까지 8방향 인접 워크로 읽어낼 수 있게 만드는
문제입니다 (문제 조건: [블로그 풀이](https://velog.io/@spark1130/%EB%B0%B1%EC%A4%80-814-2-%EB%AC%B8%EC%A0%9C-%ED%92%80%EC%9D%B4)).
점수 = 1..K가 전부 형성 가능한 최대 K (이론 최대 8142).

## 디렉터리 구조

```
Project814/
├── school_linux_backup/   Linux/Windows 개발 환경 백업 — 현재 유지보수 중인 코드
│                          (원본 DEAP GA + code/, data/, 그리고 새로 작성한 sa814/ SA 솔버)
└── hpc_linux_backup/      HPC 클러스터 환경 백업 — GPU/Jupyter 기반 초기 실험 기록
```

**현재 작업은 [`school_linux_backup/`](school_linux_backup/README.md)에서 진행 중입니다.**
그 아래 `sa814/`가 최신 솔버(시뮬레이티드 어닐링 + 병렬 템퍼링, 실제 체크포인트 지원)이고,
`code/`의 원본 DEAP GA는 비교 기준선으로 남아 있습니다. 실행 방법, 설계 근거, 검증 결과는
[school_linux_backup/README.md](school_linux_backup/README.md)와
[school_linux_backup/sa814/README.md](school_linux_backup/sa814/README.md)를 참고하세요.

`hpc_linux_backup/`은 HPC 클러스터에서 GPU/Jupyter 노트북으로 실험하던 더 이전 버전의
백업입니다.
