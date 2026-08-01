"""
Windows platform layer for sa814.

Handles the parts that differ from Linux: Windows has no real SIGTERM (a
process killed via TerminateProcess never runs Python handlers at all --
there is nothing we can do about that from within the process), but Ctrl-C
(SIGINT) and Ctrl-Break (SIGBREAK, console-only) are both catchable, and an
`atexit` hook covers normal interpreter shutdown. All paths are anchored off
this file's location rather than assuming cwd == code/, which is what the
original 814_cpu_*.py scripts required (`'../data/...'` relative paths).
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
from pathlib import Path

SA814_DIR = Path(__file__).resolve().parent
REPO_ROOT = SA814_DIR.parent

_stop_requested = False
_on_stop_callbacks = []


def should_stop() -> bool:
    return _stop_requested


def install_handlers(on_stop) -> None:
    """on_stop is called at most once, the first time a stop is requested
    (via Ctrl-C, Ctrl-Break, or normal process exit)."""
    global _stop_requested
    _on_stop_callbacks.append(on_stop)

    def _handler(signum, frame):
        global _stop_requested
        if not _stop_requested:
            sys.stderr.write(f"\n[sa814] received signal {signum}, finishing current block "
                             f"and checkpointing ...\n")
        _stop_requested = True
        for cb in _on_stop_callbacks:
            cb()

    signal.signal(signal.SIGINT, _handler)
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signal.signal(sigbreak, _handler)

    def _atexit_handler():
        global _stop_requested
        if not _stop_requested:
            _stop_requested = True
            for cb in _on_stop_callbacks:
                cb()

    atexit.register(_atexit_handler)


def default_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, n)


def base_dir() -> Path:
    return SA814_DIR


def data_dir() -> Path:
    return REPO_ROOT / "data"
