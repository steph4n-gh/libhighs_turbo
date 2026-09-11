"""Graph cutting planes, rational cut verification, and SciPy-style solvers."""

from __future__ import annotations

from highs_turbo.api import (
    MaxCutResult,
    QuboResult,
    TurboSolver,
    linprog,
    solve_maxcut,
    solve_qubo,
)
from highs_turbo.detector import TopologyDetector, TopologyScanResult, detect_topology

__version__ = "0.1.0"

__all__ = [
    "linprog",
    "solve_maxcut",
    "solve_qubo",
    "TurboSolver",
    "MaxCutResult",
    "QuboResult",
    "TopologyDetector",
    "TopologyScanResult",
    "detect_topology",
]
