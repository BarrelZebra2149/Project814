"""
Linux platform layer for sa814.

Handles SIGINT/SIGTERM/SIGHUP (nohup/systemd-friendly), optional `nice`
adjustment, and auto-resume by default (a service restart or a nohup process
getting SIGHUP on logout should just pick up the last checkpoint). All paths
are anchored off this file's location, not cwd, unlike the original
814_cpu_*.py scripts which hard-required `cwd == code/` because of their
`'../data/...'` relative paths.
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

    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handler)

    def _atexit_handler():
        global _stop_requested
        if not _stop_requested:
            _stop_requested = True
            for cb in _on_stop_callbacks:
                cb()

    atexit.register(_atexit_handler)


def apply_nice(level: int) -> None:
    if level:
        try:
            os.nice(level)
        except OSError as exc:
            sys.stderr.write(f"[sa814] could not set nice({level}): {exc}\n")


def default_threads() -> int:
    try:
        # Prefer the process's actual CPU affinity (accurate under cgroups /
        # taskset), falling back to the logical count.
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 4)


def base_dir() -> Path:
    return SA814_DIR


def data_dir() -> Path:
    return REPO_ROOT / "data"
