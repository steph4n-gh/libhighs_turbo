"""Small, exhaustive checks of the public solver's mathematical contracts."""

import itertools
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog as scipy_linprog

import highs_turbo
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance


@pytest.mark.parametrize("integrality, expected", [([], -2.5), ([1] * 5, -2.)])
def test_legacy_array_solver_and_ising_share_runtime(integrality, expected):
    from highs_turbo.compiled_engine import solve_milp_with_highs

    # The stable-set relaxation of a five-cycle is fractional; integer search
    # must close its gap, including after another API has used the runtime.
    for _ in range(2):
        result = solve_milp_with_highs(
            [-1.] * 5, [0.] * 5, [1.] * 5, [-np.inf] * 5, [1.] * 5,
            [0, 2, 4, 6, 8], [0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1.] * 10, integrality,
        )
        assert result["fun"] == pytest.approx(expected)
        assert sum(result["x"]) == pytest.approx(-expected)
        ising = highs_turbo.solve_ising({0: 2, 1: 1}, {(0, 1): -2}, accelerate=False)
        assert ising.success and ising.energy == -5


@pytest.mark.parametrize("n", [3, 4, 5])
def test_every_integer_cut_satisfies_relaxation(n):
    edges = list(itertools.combinations(range(n), 2))
    graph = GraphInstance(name="complete", num_nodes=n, edges=edges)
    _, matrix, rhs, _, _ = ExactMaxCutSolver().build_relaxation_matrices(
        graph, include_k5=True
    )
    for partition in itertools.product((0, 1), repeat=n):
        cut = np.array([partition[u] != partition[v] for u, v in edges], dtype=float)
        assert np.all(matrix @ cut <= rhs)


def test_signed_triangle_relaxation_does_not_exclude_optimum():
    graph = GraphInstance(
        name="signed_triangle", num_nodes=3,
        edges=[(0, 1), (0, 2), (1, 2)],
        weights={(0, 1): 1.0, (0, 2): 1.0, (1, 2): -10.0},
    )
    solver = ExactMaxCutSolver()
    optimum, _ = solver.solve_integer_maxcut(graph)
    relaxation = solver.solve_cycle_relaxation(graph)
    assert optimum == 2.0
    assert relaxation.objective_value >= optimum


@pytest.mark.parametrize("Q", [
    np.array([[0.0, -1.0], [-1.0, 0.0]]),
    np.array([[-0.0000001, 0.0], [0.0, 0.3]]),
    np.array([[0.1, -0.7, 1.0], [-0.2, 0.2, -0.3], [0.0, 0.5, -0.1]]),
])
def test_qubo_bound_is_below_every_exact_binary_energy(Q):
    result = highs_turbo.solve_qubo(Q)
    energies = []
    for state in itertools.product((0, 1), repeat=len(Q)):
        energy = sum((
            Fraction.from_float(float(Q[i, j])) * state[i] * state[j]
            for i in range(len(Q)) for j in range(len(Q))
        ), Fraction(0))
        energies.append(energy)
        assert result.exact_rational_bound <= energy
    assert result.energy == pytest.approx(float(min(energies)))
    assert result.status == "OPTIMAL"
    assert result.simplex_iterations == 0


def test_qubo_receipt_includes_objective_coefficients():
    first = highs_turbo.solve_qubo(np.diag([-1.0, 2.0]))
    second = highs_turbo.solve_qubo(np.diag([-1.0, 3.0]))
    assert first.lower_bound == second.lower_bound
    assert first.certificate != second.certificate


@pytest.mark.parametrize("Q", [{}, np.empty((0, 0)), sparse.csr_matrix((0, 0))])
def test_empty_qubo(Q):
    result = highs_turbo.solve_qubo(Q)
    assert result.energy == result.lower_bound == 0.0
    assert result.solution.size == 0


@pytest.mark.parametrize("Q", [1.0, [1.0, 2.0], np.zeros((2, 3)), [[np.nan]], [[np.inf]]])
def test_invalid_qubo_raises_value_error(Q):
    with pytest.raises(ValueError):
        highs_turbo.solve_qubo(Q)


@pytest.mark.parametrize("n", [0, 3])
def test_integer_maxcut_with_no_edges(n):
    graph = GraphInstance(name="empty", num_nodes=n, edges=[])
    value, partition = ExactMaxCutSolver().solve_integer_maxcut(graph)
    assert value == 0
    np.testing.assert_array_equal(partition, np.zeros(n, dtype=int))


@pytest.mark.parametrize("key", ["graph", "adj", "adjacency"])
@pytest.mark.parametrize("as_sparse", [False, True])
def test_explicit_adjacency_does_not_require_truth_value(key, as_sparse):
    adjacency = np.ones((3, 3)) - np.eye(3)
    if as_sparse:
        adjacency = sparse.csr_matrix(adjacency)
    scan = highs_turbo.detect_topology([-1.0] * 3, **{key: adjacency})
    assert scan.is_graph_structured
    assert scan.graph.num_edges == 3


def test_explicit_empty_networkx_graph_is_not_ignored():
    scan = highs_turbo.detect_topology([], graph=nx.Graph())
    assert scan.is_graph_structured
    assert scan.graph.num_nodes == 0


def test_local_search_results_are_labelled_heuristic():
    assert highs_turbo.solve_qubo(np.eye(19)).status == "HEURISTIC"
    assert highs_turbo.solve_maxcut(nx.path_graph(201)).status == "HEURISTIC"


@pytest.mark.parametrize("integer_coefficients", [False, True])
def test_accelerated_lp_preserves_primal_and_dual_solution(integer_coefficients):
    rng = np.random.default_rng(18)
    A = rng.integers(0, 5, (800, 20)).astype(float)
    if not integer_coefficients:
        A += rng.random(A.shape)
    b = A @ np.ones(20) + rng.integers(0, 10, 800)
    c = -rng.random(20)
    eq = np.ones((1, 20))
    kwargs = dict(A_ub=sparse.csr_matrix(A), b_ub=b, A_eq=eq, b_eq=[18], bounds=(0, 2))
    result = highs_turbo.linprog(c, **kwargs)
    reference = scipy_linprog(c, **kwargs)
    assert result.turbo_accelerated
    assert result.fun == pytest.approx(reference.fun, abs=1e-7)
    assert np.min(result.slack) >= -1e-7
    np.testing.assert_allclose(result.con, 0, atol=1e-7)
    np.testing.assert_allclose(result.slack, b - A @ result.x, atol=1e-9)
    dual_residual = (c - A.T @ result.ineqlin.marginals - eq.T @ result.eqlin.marginals
                     - result.lower.marginals - result.upper.marginals)
    np.testing.assert_allclose(dual_residual, 0, atol=1e-7)
    np.testing.assert_allclose(result.slack * result.ineqlin.marginals, 0, atol=1e-7)
    dual_objective = (b @ result.ineqlin.marginals + 18 * result.eqlin.marginals[0]
                      + 2 * result.upper.marginals.sum())
    assert dual_objective == pytest.approx(result.fun, abs=1e-7)
    if integer_coefficients:
        assert result.turbo_surrogate_rows > 0


@pytest.mark.parametrize("variable_type", [1, 2, 3])
def test_integer_and_semicontinuous_models_keep_native_mip_semantics(variable_type):
    rng = np.random.default_rng(12)
    A = rng.integers(0, 5, (600, 8)).astype(float)
    b = A.sum(axis=1) + rng.integers(0, 5, 600)
    c = -rng.random(8)
    kwargs = dict(A_ub=A, b_ub=b, bounds=(1, 2), integrality=variable_type)
    result = highs_turbo.linprog(c, **kwargs)
    reference = scipy_linprog(c, **kwargs)
    assert result.success
    assert result.fun == pytest.approx(reference.fun, abs=1e-7)
    assert np.max(A @ result.x - b) <= 1e-7
    assert result.mip_gap <= 1e-4
    assert np.all((result.x >= 1 - 1e-7) | (np.abs(result.x) < 1e-7))
    if variable_type != 2:
        np.testing.assert_allclose(result.x, np.round(result.x), atol=1e-7)


@pytest.mark.parametrize("kind", ["unbounded", "infeasible", "budget"])
def test_original_status_and_explicit_budget_are_preserved(kind):
    A = np.ones((600, 2))
    c = [-1, 0]
    kwargs = dict(A_ub=A, b_ub=np.ones(600))
    if kind == "unbounded":
        A[:, 0] = 0
    elif kind == "infeasible":
        kwargs["b_ub"][0] = -1
    else:
        kwargs["options"] = {"time_limit": 0}
    result = highs_turbo.linprog(c, **kwargs)
    reference = scipy_linprog(c, **kwargs)
    assert result.status == reference.status
    assert result.success == reference.success
    assert result.x is None
    assert not result.turbo_accelerated


@pytest.mark.parametrize("mode", ["direct", "portfolio", "primary_error"])
def test_sparse_native_solve_preserves_primal_dual_and_joins_workers(mode, monkeypatch):
    import threading
    import time
    import highspy

    rng = np.random.default_rng(53)
    rows, cols = (40, 24) if mode == "direct" else (600, 1100)
    A = sparse.random(rows, cols, density=0.04, random_state=rng, format="csr")
    eq = sparse.random(20, cols, density=0.04, random_state=rng, format="csr")
    bounds = np.tile([0.0, 2.0], (cols, 1))
    bounds[0] = [-np.inf, np.inf]
    bounds[1] = [0.5, 0.5]
    point = np.ones(cols)
    point[1] = 0.5
    b, d, c = A @ point + rng.random(rows), eq @ point, -rng.random(cols)
    # Ensure the free variable participates in a bounding equality.
    eq = eq.tolil()
    eq[0, 0] = 1.0
    eq = eq.tocsr()
    d = eq @ point
    kwargs = dict(A_ub=A, b_ub=b, A_eq=eq, b_eq=d, bounds=bounds)
    reference = scipy_linprog(c, **kwargs)
    monkeypatch.setattr("highs_turbo.lp_accelerator.os.cpu_count", lambda: 2)
    if mode == "primary_error":
        original_run = highspy.Highs.run

        def fail_simplex(session):
            if session.getOptionValue("solver")[1] == "simplex":
                time.sleep(0.01)  # Allow the independent method to start.
                raise RuntimeError("Injected simplex failure")
            return original_run(session)

        monkeypatch.setattr(highspy.Highs, "run", fail_simplex)

    before = set(threading.enumerate())
    result = highs_turbo.TurboSolver(fallback_on_error=False).linprog(c, **kwargs)
    assert not (set(threading.enumerate()) - before)
    assert result.success and reference.success
    assert result.fun == pytest.approx(reference.fun, abs=1e-7)
    assert np.min(result.slack) >= -1e-7
    np.testing.assert_allclose(result.con, 0, atol=1e-7)
    np.testing.assert_allclose(
        c - A.T @ result.ineqlin.marginals - eq.T @ result.eqlin.marginals
        - result.lower.marginals - result.upper.marginals, 0, atol=1e-7,
    )
    assert result.nit == sum(result.turbo_solver_iterations.values())
    if mode == "primary_error":
        assert result.turbo_strategy == "highs_portfolio"
        assert result.turbo_solver == "ipm"
