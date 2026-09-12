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
# Exercise both acceleration paths from the installed package, including when
# the optional C++ cut engine is absent.
from scipy.optimize import linprog
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import generate_k5_cluster_graph

graph = generate_k5_cluster_graph(16, 15)
c, matrix, rhs, _, _ = ExactMaxCutSolver().build_relaxation_matrices(graph)
certified = highs_turbo.linprog(c, A_ub=matrix, b_ub=rhs, bounds=(0, 1))
assert certified.success and certified.turbo_strategy == "conic_certificate"
np.testing.assert_allclose(certified.fun, linprog(c, A_ub=matrix, b_ub=rhs, bounds=(0, 1)).fun)

rng = np.random.default_rng(4)
matrix = rng.random((600, 20))
rhs = matrix.sum(axis=1) + rng.integers(0, 5, 600)
c = -rng.random(20)
recovered = highs_turbo.linprog(c, A_ub=matrix, b_ub=rhs, bounds=(0, 2))
assert recovered.success and recovered.turbo_strategy == "row_recovery"
assert np.max(matrix @ recovered.x - rhs) <= 1e-7
np.testing.assert_allclose(recovered.fun, linprog(c, A_ub=matrix, b_ub=rhs, bounds=(0, 2)).fun)

assert highs_turbo.solve_maxcut(np.ones((3, 3)) - np.eye(3)).cut_value == 2.0
qubo = highs_turbo.solve_qubo([[0.0, -1.0], [-1.0, 0.0]])
assert qubo.energy == qubo.lower_bound == -2.0

# Exercise both MIP entry points in the same installed process. Loading a
# separate system HiGHS in the graph extension previously crashed on macOS.
from highs_turbo.compiled_engine import solve_milp_with_highs

legacy = solve_milp_with_highs(
    [-1.] * 5, [0.] * 5, [1.] * 5, [-np.inf] * 5, [1.] * 5,
    [0, 2, 4, 6, 8, 10], [0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1.] * 10, [1] * 5,
)
assert legacy["fun"] == -2 and sum(legacy["x"]) == 2
ising = highs_turbo.solve_ising({0: 2, 1: 1, 2: 0}, {(0, 1): -2, (0, 2): -3}, accelerate=False)
assert ising.status == "OPTIMAL" and ising.energy == ising.lower_bound == -8
complete = {(i, j): 1. for i in range(6) for j in range(i+1, 6)}
verified = highs_turbo.solve_ising({}, complete, cut_policy="learned", certified_gap=0, time_limit=5)
assert verified.exact_gap == 0 and verified.energy == -3
assert highs_turbo.verify_ising_certificate({}, complete, verified.certificate.to_dict())
global_bound = highs_turbo.solve_ising({}, complete, relaxation="sdp", certified_gap=0, time_limit=5)
assert global_bound.exact_gap == 0 and global_bound.certificate.gram_factor
assert highs_turbo.verify_ising_certificate({}, complete, global_bound.certificate.to_dict())
# Read the same square through the sparse on-disk format in the built wheel.
sparse_receipt = global_bound.certificate.to_dict()
rows = sparse_receipt["gram_factor"]
starts = [0]
for row in rows:
    starts.append(starts[-1] + len(row))
sparse_receipt.update(version=4, gram_factor=[], sparse_gram_factor=[
    starts, [j for row in rows for j in range(len(row))], [x for row in rows for x in row]])
assert highs_turbo.verify_ising_certificate({}, complete, sparse_receipt)

# The drop-in product bridge and standalone checker must ship in the wheel.
c = np.r_[np.ones(32), -3.]
envelope = np.zeros((3, 33))
envelope[0, [32, 0]] = [1, -1]
envelope[1, [32, 1]] = [1, -1]
envelope[2, [0, 1, 32]] = [1, 1, -1]
product = highs_turbo.linprog(c, A_ub=envelope, b_ub=[0, 0, 1],
                            bounds=(0, 1), integrality=np.r_[np.ones(32), 0])
assert product.success and product.fun == -1 and product.turbo_equivalent_model
assert highs_turbo.verify_linprog_certificate(
    c, product.turbo_bound_certificate, A_ub=envelope, b_ub=[0, 0, 1],
    bounds=(0, 1), integrality=np.r_[np.ones(32), 0])

if "--require-native" in sys.argv:
    assert COMPILED_ENGINE_AVAILABLE
if "--require-fallback" in sys.argv:
    assert not COMPILED_ENGINE_AVAILABLE
if "--require-ml" in sys.argv:
    from highs_turbo.surrogate_model import EdgeEquivariantSurrogateGNN

    assert isinstance(_DEFAULT_SOLVER.gnn_model, EdgeEquivariantSurrogateGNN)
print(f"Installed package smoke checks passed (native={COMPILED_ENGINE_AVAILABLE}).")
