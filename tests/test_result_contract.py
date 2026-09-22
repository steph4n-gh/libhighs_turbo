"""Proof status, exact objectives, and legacy result compatibility."""

from dataclasses import replace
from fractions import Fraction
import itertools

import networkx as nx
import numpy as np
import pytest

import highs_turbo
from highs_turbo.api import MaxCutResult, QuboResult


@pytest.mark.parametrize("kind", ["ising", "qubo", "maxcut"])
def test_exact_objective_and_gap_use_original_problem(kind):
    if kind == "ising":
        fields = [Fraction(1, 3), Fraction(-1, 7), 0]
        edges = {(0, 1): Fraction(2, 3), (1, 2): -2, (0, 2): 1}
        offset = Fraction(1, 5)
        result = highs_turbo.solve_ising(fields, edges, offset=offset, time_limit=0)
        objective = lambda state: offset + sum(h * s for h, s in zip(fields, state)) + sum(
            w * state[u] * state[v] for (u, v), w in edges.items())
        state = list(result.spins.values())
        values = [objective(s) for s in itertools.product((-1, 1), repeat=3)]
        bound = result.exact_cut_lower_bound
    elif kind == "qubo":
        matrix = np.array([[-.1, 2., -.5], [1., -2., .25], [0., 0., .3]])
        result = highs_turbo.solve_qubo(matrix, time_limit=0)
        objective = lambda state: sum((Fraction(float(matrix[i, j])) * state[i] * state[j]
                                      for i in range(3) for j in range(3)), Fraction())
        state = result.solution
        values = [objective(s) for s in itertools.product((0, 1), repeat=3)]
        bound = result.exact_rational_bound
    else:
        edges = {(0, 1): Fraction(1, 3), (1, 2): -2, (0, 2): 1}
        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        graph.add_weighted_edges_from((u, v, w) for (u, v), w in edges.items())
        result = highs_turbo.solve_maxcut(graph, time_limit=0)
        objective = lambda state: sum((w for (u, v), w in edges.items()
                                      if state[u] != state[v]), Fraction())
        state = result.partition
        values = [objective(s) for s in itertools.product((0, 1), repeat=3)]
        bound = result.exact_rational_bound
    assert result.exact_objective == objective(state)
    assert isinstance(result.exact_objective, Fraction)
    expected_gap = bound - objective(state) if kind == "maxcut" else objective(state) - bound
    assert result.exact_gap == expected_gap >= 0
    assert bound >= max(values) if kind == "maxcut" else bound <= min(values)
    assert result.bound_verified
    assert result.optimality_proven == (expected_gap == 0)


@pytest.mark.parametrize("kind", ["ising", "qubo", "maxcut"])
def test_exact_optimality_is_independent_of_solver_status(kind):
    if kind == "ising":
        result = highs_turbo.solve_ising([-Fraction(1, 3)], {})
    elif kind == "qubo":
        result = highs_turbo.solve_qubo([[-.1]])
    else:
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=Fraction(1, 3))
        result = highs_turbo.solve_maxcut(graph)
    assert result.bound_verified and result.exact_gap == 0 and result.optimality_proven
    assert replace(result, status="TIME_LIMIT").optimality_proven
    assert not replace(result, bound_verified=False).optimality_proven


@pytest.mark.parametrize("kind", ["qubo", "maxcut"])
@pytest.mark.parametrize("solver_status", ["TIME_LIMIT", "INTERRUPTED", "SOLVER_ERROR", "OPTIMAL"])
def test_wrappers_preserve_solver_diagnostics_without_claiming_exact_optimality(
        monkeypatch, kind, solver_status):
    import highs_turbo.ising as ising

    original = ising.solve_ising

    def reported_result(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.bound_verified and result.exact_gap > 0
        return replace(result, status=solver_status, message="injected termination",
                       certificate_fallbacks=("injected proof fallback",))

    monkeypatch.setattr(ising, "solve_ising", reported_result)
    result = (highs_turbo.solve_qubo([[-4, 4, 4], [0, -4, 4], [0, 0, -4]], time_limit=0)
              if kind == "qubo" else highs_turbo.solve_maxcut(nx.complete_graph(3), time_limit=0))
    assert result.solver_status == solver_status
    assert result.message == "injected termination"
    assert result.certificate_fallbacks == ("injected proof fallback",)
    assert result.status == ("OPTIMAL" if solver_status == "OPTIMAL" else "HEURISTIC")
    assert result.is_rationally_certified and result.bound_verified
    assert result.exact_gap > 0 and not result.optimality_proven


@pytest.mark.parametrize("kind", ["qubo", "maxcut"])
def test_wrapper_legacy_tuple_summary_and_manual_defaults(kind):
    if kind == "qubo":
        result = highs_turbo.solve_qubo([[-1.]])
        objective_name, answer_name, bound_name = "energy", "solution", "lower_bound"
        manual = QuboResult(-1., np.array([1]), "receipt", exact_rational_bound=Fraction(-1),
                            exact_objective=Fraction(-1))
    else:
        result = highs_turbo.solve_maxcut(nx.path_graph(2))
        objective_name, answer_name, bound_name = "cut_value", "partition", "upper_bound"
        manual = MaxCutResult(1., np.array([0, 1]), "receipt", exact_rational_bound=Fraction(1),
                              exact_objective=Fraction(1))
    assert len(result) == 3
    objective, answer, receipt = result
    assert objective == getattr(result, objective_name) == result[0]
    np.testing.assert_array_equal(answer, getattr(result, answer_name))
    assert receipt == result.certificate == result[2]
    assert set(result.to_dict()) == {
        objective_name, answer_name, bound_name, "certificate", "is_rationally_certified",
        "simplex_iterations", "solve_time_ms", "status"}
    assert manual.is_rationally_certified and manual.exact_gap == 0
    assert not manual.bound_verified and not manual.optimality_proven
    assert replace(manual, exact_objective=None).exact_gap is None
    assert not replace(manual, exact_objective=None, bound_verified=True).optimality_proven


@pytest.mark.parametrize("kind", ["ising", "qubo", "maxcut"])
def test_empty_models_have_exact_zero_proof_status(kind):
    result = (highs_turbo.solve_ising([], {}) if kind == "ising" else
              highs_turbo.solve_qubo(np.zeros((0, 0))) if kind == "qubo" else
              highs_turbo.solve_maxcut(nx.empty_graph(3)))
    assert result.exact_objective == result.exact_gap == 0
    assert result.bound_verified and result.optimality_proven
    if kind == "maxcut":
        # Preserve the established empty-Max-Cut receipt/witness convention.
        assert result.certificate == "0" * 64 and result.bound_certificate is None
