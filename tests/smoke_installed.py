"""Run with `python -I tests/smoke_installed.py` after installing a built wheel."""

import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

import highs_turbo
from highs_turbo.api import _DEFAULT_SOLVER
from highs_turbo.compiled_engine import COMPILED_ENGINE_AVAILABLE
from highs_turbo.large_scale_benchmarks import generate_chimera_instance


source_package = Path(__file__).resolve().parents[1] / "highs_turbo"
assert Path(highs_turbo.__file__).resolve().parent != source_package
assert files("highs_turbo").joinpath("default_weights.pt").is_file()
assert generate_chimera_instance(1).num_nodes == 8
assert highs_turbo.linprog([-1.0], bounds=(0, 1)).fun == -1.0
assert highs_turbo.solve_maxcut(np.ones((3, 3)) - np.eye(3)).cut_value == 2.0
qubo = highs_turbo.solve_qubo([[0.0, -1.0], [-1.0, 0.0]])
assert qubo.energy == qubo.lower_bound == -2.0

if "--require-native" in sys.argv:
    assert COMPILED_ENGINE_AVAILABLE
if "--require-fallback" in sys.argv:
    assert not COMPILED_ENGINE_AVAILABLE
if "--require-ml" in sys.argv:
    from highs_turbo.surrogate_model import EdgeEquivariantSurrogateGNN

    assert isinstance(_DEFAULT_SOLVER.gnn_model, EdgeEquivariantSurrogateGNN)
print(f"Installed package smoke checks passed (native={COMPILED_ENGINE_AVAILABLE}).")
