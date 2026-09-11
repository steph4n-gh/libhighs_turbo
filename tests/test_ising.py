"""Check original energies, cut validity, and bounds against exhaustive states."""

from fractions import Fraction
from itertools import product

import networkx as nx
import numpy as np
import pytest
from dwave.graphs import pegasus_graph

from highs_turbo import TurboSolver, solve_ising
from highs_turbo.ising import _cut_data
from highs_turbo.topologies import generate_pegasus_instance


def energy(h, J, state, offset=0):
    return (Fraction(float(offset))
            + sum(Fraction(float(w)) * state[i] for i, w in h.items())
            + sum(Fraction(float(w)) * state[i] * state[j] for (i, j), w in J.items()))


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_ising_matches_exhaustive_pegasus_with_fields(seed):
    graph = pegasus_graph(3)
    labels = list(nx.bfs_tree(graph, next(iter(graph))))[:10]
    rng = np.random.default_rng(seed)
    h = {i: float(rng.integers(-3, 4)) / 8 for i in labels}
    J = {edge: float(rng.integers(-8, 9)) / 8 for edge in graph.subgraph(labels).edges}
    optimum = min(energy(h, J, dict(zip(labels, state)), 0.3)
                  for state in product([-1, 1], repeat=len(labels)))
    for accelerated in (False, True):
        result = solve_ising(h, J, offset=0.3, accelerate=accelerated)
        assert result.status == "OPTIMAL"
        assert result.exact_energy == energy(h, J, result.spins, 0.3) == optimum
        assert result.exact_cut_lower_bound <= optimum
        assert result.lower_bound <= float(optimum) + 1e-7
        assert result.gap <= 1e-7


def test_ising_cut_rows_and_conic_bound_are_valid_for_every_state():
    graph = nx.cycle_graph(5)
    graph.add_edges_from([(0, 2), (1, 3)])
    edges = sorted(graph.edges)
    weights = {edge: Fraction((-1)**i * (i + 1), 8) for i, edge in enumerate(edges)}
    rows, rhs, lower, certificate = _cut_data(graph, edges, weights, Fraction(1, 3), float("inf"))
    assert certificate.is_valid
    for state in product([-1, 1], repeat=5):
        x = np.array([state[u] != state[v] for u, v in edges], dtype=float)
        assert np.all(rows @ x <= rhs)
        assert lower <= Fraction(1, 3) + sum(w * state[u] * state[v] for (u, v), w in weights.items())


def test_ising_exact_certificate_and_original_labels():
    h = {"a": 0, (2, 3): 0, 99: 0, "isolated": 0}
    J = {("a", (2, 3)): 0.5, ((2, 3), "a"): 0.5,
         ((2, 3), 99): 1, (99, "a"): 1, ("a", "a"): 0.25}
    result = TurboSolver().solve_ising(h, J, offset=0.5)
    assert result.status == "OPTIMAL"
    assert result.is_rationally_certified
    assert result.exact_cut_lower_bound == result.exact_energy == Fraction(-1, 4)
    assert set(result.spins) == set(h)
    assert result.exact_energy == energy(h, J, result.spins, 0.5)


def test_zero_budget_keeps_a_feasible_state_and_bound(monkeypatch):
    import highspy
    monkeypatch.setattr(highspy.Highs, "run", lambda self: pytest.fail("No search budget"))
    J = {(0, 1): 1, (1, 2): 1, (0, 2): 1}
    result = solve_ising({}, J, time_limit=0)
    assert result.status == "TIME_LIMIT"
    assert energy({}, J, result.spins) == result.exact_energy
    assert result.lower_bound <= -1 <= result.energy
    assert result.gap == result.energy - result.lower_bound
    assert not result.is_rationally_certified


def test_requested_relative_gap_does_not_claim_a_closed_proof():
    graph = pegasus_graph(4)
    selected = list(nx.bfs_tree(graph, list(graph)[7]))[:32]
    rng = np.random.default_rng(1)
    J = {edge: int(rng.choice([-1, 1])) for edge in sorted(graph.subgraph(selected).edges)}
    result = solve_ising({i: 0 for i in selected}, J, accelerate=False, relative_gap=10)
    assert result.status == ("GAP_LIMIT" if result.gap > 1e-6 else "OPTIMAL")
    assert result.energy == energy({}, J, result.spins)


@pytest.mark.parametrize("kwargs", [dict(h={0: np.nan}, J={}), dict(h={}, J={(0, 1): np.inf}),
                                    dict(h={}, J={}, offset=np.inf), dict(h={}, J={}, time_limit=-1)])
def test_ising_rejects_invalid_numbers(kwargs):
    with pytest.raises(ValueError):
        solve_ising(**kwargs)


def test_empty_and_diagonal_ising():
    assert solve_ising({}, {}, offset=2.25).exact_energy == Fraction(9, 4)
    result = solve_ising([2, -3], {(0, 0): 0.5})
    assert result.spins == {0: -1, 1: 1}
    assert result.energy == result.lower_bound == -4.5


def test_pegasus_is_official_fabric_and_preserves_subgraph_labels():
    official = pegasus_graph(4)
    graph = generate_pegasus_instance(4)
    labels = graph.metadata["node_labels"]
    assert set(labels) == set(official)
    assert {frozenset((labels[u], labels[v])) for u, v in graph.edges} == {frozenset(e) for e in official.edges}
    assert max(dict(graph.to_networkx().degree()).values()) <= 15
    chosen = list(nx.bfs_tree(official, next(iter(official))))[:12]
    couplers = list(official.subgraph(chosen).edges)[::2]
    subgraph = generate_pegasus_instance(4, node_list=chosen, edge_list=couplers)
    labels = subgraph.metadata["node_labels"]
    assert set(labels) == set(chosen)
    assert {frozenset((labels[u], labels[v])) for u, v in subgraph.edges} == {frozenset(e) for e in couplers}
    invalid = next((u, v) for u in chosen for v in chosen if u != v and not official.has_edge(u, v))
    with pytest.raises(ValueError):
        generate_pegasus_instance(4, node_list=chosen, edge_list=[invalid])
