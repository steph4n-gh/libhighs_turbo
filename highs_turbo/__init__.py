"""Graph cutting planes, rational cut verification, and SciPy-style solvers."""

from __future__ import annotations

# Initialize the public HiGHS runtime before the optional extension, which can
# link another system HiGHS. On macOS the reverse order breaks MIP scheduling.
import highspy as _highspy

from highs_turbo.api import (
    MaxCutResult,
    QuboResult,
    TurboSolver,
    linprog,
    solve_maxcut,
    solve_qubo,
)
from highs_turbo.detector import TopologyDetector, TopologyScanResult, detect_topology
from highs_turbo.ising import IsingResult, solve_ising

__version__ = "0.1.0"

__all__ = [
    "linprog",
    "solve_maxcut",
    "solve_qubo",
    "solve_ising",
    "IsingResult",
    "TurboSolver",
    "MaxCutResult",
    "QuboResult",
    "TopologyDetector",
    "TopologyScanResult",
    "detect_topology",
]
