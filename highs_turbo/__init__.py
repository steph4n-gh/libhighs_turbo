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
from highs_turbo.ising import IsingResult, solve_ising, verify_ising_certificate
from highs_turbo.ising_cuts import IsingCertificate
from highs_turbo.ising_linearized import verify_linprog_certificate

__version__ = "0.3.0"

from highs_turbo.certification import certify, verify

__all__ = [
    "certify",
    "verify",
    "linprog",
    "solve_maxcut",
    "solve_qubo",
    "solve_ising",
    "IsingResult",
    "IsingCertificate",
    "verify_ising_certificate",
    "verify_linprog_certificate",
    "TurboSolver",
    "MaxCutResult",
    "QuboResult",
    "TopologyDetector",
    "TopologyScanResult",
    "detect_topology",
]
