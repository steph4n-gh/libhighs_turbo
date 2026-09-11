"""Small, exhaustive checks of the public solver's mathematical contracts."""

import itertools
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

import highs_turbo
from highs_turbo.exact_solver import ExactMaxCutSolver
from highs_turbo.graph_generator import GraphInstance


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
