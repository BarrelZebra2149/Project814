"""Windows entry point: maximize the real consecutive score (814-2 primary
objective) via simulated annealing / parallel tempering.

    python win_score_first.py --seconds 600
    python win_score_first.py --resume
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runtime_win as runtime
from config814 import config_from_cli
from driver import drive


def main():
    cfg = config_from_cli(default_preset="score_first")
    if cfg.run_name == "default":
        cfg.run_name = "win_score_first"
    drive(cfg, runtime, runtime.base_dir())


if __name__ == "__main__":
    main()
