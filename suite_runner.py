"""
suite_runner.py -- run several test_phaseN.py files concurrently, each in its own subprocess with a hard timeout.

Used by test_phase4 / test_phase5 to check that the earlier phases still pass.  The suites are independent (each
writes only into its own temporary directory and only reads the shared data files), so running them in parallel is
safe and deterministic; a hung suite is killed after ``timeout`` seconds and reported by name instead of stalling.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

TESTS = Path(__file__).resolve().parent
DEFAULT_TIMEOUT = 300          # seconds per suite
PARENT_ENV = "AUDITTAILOR_SUITES_RUN_BY_PARENT"


# Single-threaded numeric libraries: several suites run concurrently, and each imports numpy / scikit-learn.  With
# the default (one BLAS/OpenMP thread per core, per process) the processes oversubscribe the CPU and crawl.
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def single_thread_env() -> dict:
    return {**os.environ, **{k: os.environ.get(k, v) for k, v in THREAD_ENV.items()}}


def nested_run_by_parent() -> bool:
    """True when a parent suite is already running Phase 1..N explicitly (so a child need not repeat them)."""
    return os.environ.get(PARENT_ENV) == "1"


def run_suites(numbers: Iterable[int], timeout: float = DEFAULT_TIMEOUT) -> None:
    """Assert that ``tests/test_phase<N>.py`` exits 0 and prints ``ALL PHASE <N> TESTS PASSED`` for every N."""
    env = {**single_thread_env(), PARENT_ENV: "1"}
    procs = {n: subprocess.Popen([sys.executable, str(TESTS / f"test_phase{n}.py")],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for n in numbers}
    failures = []
    for n, p in procs.items():
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            failures.append(f"phase {n} TIMED OUT after {timeout:g}s\n{out[-500:]}\n{err[-500:]}")
            continue
        if p.returncode != 0 or f"ALL PHASE {n} TESTS PASSED" not in out:
            failures.append(f"phase {n} failed (rc={p.returncode}):\n{out[-800:]}\n{err[-800:]}")
    assert not failures, "\n\n".join(failures)
