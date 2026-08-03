"""Linux entry point: run the seed tournament orchestrator (tournament.py).

    python3 linux_tournament.py --name prod_tourney_A --pop-size 6 --replicas 12
    nohup python3 linux_tournament.py --name prod_tourney_A --pop-size 6 &
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tournament import main


if __name__ == "__main__":
    main()
