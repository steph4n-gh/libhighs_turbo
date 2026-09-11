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


def test_adaptive_subgraph_cut_closes_gap_beyond_cycle_relaxation():
    from highs_turbo import verify_ising_certificate
    # The uniform K6 cycle relaxation has x=2/3 and energy -5; the true
    # maximum cut is 9 edges, and its Ising energy is -3.
    J = {edge: 1 for edge in nx.complete_graph(6).edges}
    result = solve_ising({}, J, certified_gap=0, time_limit=5)
    assert result.exact_energy == result.exact_cut_lower_bound == -3
    assert result.subgraph_cuts >= 1
    assert verify_ising_certificate({}, J, result.certificate.to_dict())


def test_serialized_certificate_rejects_tampering_without_a_solver(monkeypatch):
    import copy
    import highspy
    from highs_turbo import verify_ising_certificate
    J = {(0, 1): .5, (0, 2): .5, (1, 2): .5}
    result = solve_ising({}, J)
    witness = result.certificate.to_dict()
    monkeypatch.setattr(highspy.Highs, 'run', lambda self: pytest.fail('Checker must not run a solver'))
    assert verify_ising_certificate({}, J, witness)
    assert not verify_ising_certificate({}, J, witness, offset=.125)
    for key, value in [('denominator', 0), ('multipliers', [-1]), ('lower_bound', [123, 1])]:
        altered = copy.deepcopy(witness)
        altered[key] = value
        assert not verify_ising_certificate({}, J, altered)
    altered = copy.deepcopy(witness)
    altered['cuts'][0]['rhs'] -= 1
    assert not verify_ising_certificate({}, J, altered)
    assert not verify_ising_certificate({}, J, {})


def test_numerical_multipliers_receive_exact_residual_repair():
    import time
    from highs_turbo.ising_cuts import IsingCut, make_certificate, optimize_dual, check_certificate
    edges = sorted(nx.complete_graph(5).edges)
    weights = {edge: Fraction(1, 8) for edge in edges}
    cut = IsingCut(tuple(range(10)), (1,)*10, 6, 'subgraph')
    dual, point = optimize_dual([cut], np.full(10, .125), [0.], time.perf_counter()+1, 0)
    assert np.isfinite(point).all()
    # Perturb both sides of the optimum, including outside LP tolerances.
    for multiplier in [dual[0], .1249999999, .1250000001, .2]:
        proof = make_certificate([cut], [-multiplier], edges, weights, Fraction(1, 10))
        assert check_certificate(proof, edges, weights, Fraction(1, 10))
        optimum = min(Fraction(1, 10)+sum(w*s[u]*s[v] for (u,v), w in weights.items())
                      for s in product([-1, 1], repeat=5))
        assert proof.lower_bound <= optimum


def test_long_cycle_separator_and_integer_support_validation():
    import time
    from highs_turbo.ising_cuts import IsingCut, separate_cycles, verify_cut
    graph = nx.cycle_graph(7)
    edges = sorted(tuple(sorted(e)) for e in graph.edges)
    cuts = separate_cycles(graph, edges, np.ones(7), [0], time.perf_counter()+1)
    assert cuts and all(verify_cut(cut, edges) for cut in cuts)
    assert any(len(cut.indices) == 7 for cut in cuts)
    assert not verify_cut(IsingCut(tuple(range(7)), (1,)*7, 5, 'cycle'), edges)
    assert not verify_cut(IsingCut(tuple(range(7)), (1,)*7, 5, 'subgraph'), edges)


def test_learned_ranking_preserves_exact_proof_and_target_gap():
    from highs_turbo import verify_ising_certificate
    J = {edge: 1 for edge in nx.complete_graph(6).edges}
    result = solve_ising({}, J, time_limit=5, cut_policy='learned', certified_gap=0)
    assert result.exact_gap == 0
    assert verify_ising_certificate({}, J, result.certificate)
    # An absolute certificate target can stop before numerical optimality.
    loose = solve_ising({}, J, time_limit=5, certified_gap=20)
    assert loose.exact_gap <= 20
    assert verify_ising_certificate({}, J, loose.certificate)


@pytest.mark.parametrize('options', [{'cut_policy': 'unknown'}, {'certified_gap': -1},
                                    {'threads': -1}, {'threads': True}, {'seed': -1},
                                    {'relaxation': 'unknown'},
                                    {'relaxation': 'hybrid', 'cut_policy': 'static'}])
def test_adaptive_options_are_validated(options):
    with pytest.raises(ValueError):
        solve_ising({}, {}, **options)


def test_gram_bound_matches_exact_dense_arithmetic_and_all_spin_states():
    from highs_turbo.ising_sdp import gram_lower_bound
    # A sparse signed objective and a dense factor exercise nonedge fill-in.
    edges = [(0, 1), (0, 4), (1, 3), (2, 4)]
    weights = [3, -7, 2, 5]
    factor = ((3,), (-2, 4), (1, -3, 2), (4, 1, -2, 1), (-1, 2, 3, -4, 2))
    actual = gram_lower_bound(edges, weights, 8, factor, 16)
    residual = [[Fraction() for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            residual[i][j] = -Fraction(sum(factor[i][k]*factor[j][k]
                                             for k in range(min(i, j)+1)), 256)
    for (u, v), w in zip(edges, weights):
        residual[u][v] += Fraction(w, 16)
        residual[v][u] += Fraction(w, 16)
    expected = sum(residual[i][i] for i in range(5)) - sum(
        abs(residual[i][j]) for i in range(5) for j in range(5) if i != j)
    assert actual == expected
    assert all(actual <= sum(Fraction(w, 8)*s[u]*s[v] for (u,v), w in zip(edges, weights))
               for s in product([-1, 1], repeat=5))


@pytest.mark.parametrize('relaxation', ['sdp', 'hybrid'])
def test_global_relaxation_certificate_on_weighted_original_problem(relaxation, monkeypatch):
    import copy
    from highs_turbo import verify_ising_certificate
    h = {'a': .3, 7: -.5, ('b',): .75, 'isolated': 0}
    J = {('a', 7): .625, (7, ('b',)): -.75, (('b',), 'a'): .5, ('a', 'a'): .25}
    result = solve_ising(h, J, offset=.1, relaxation=relaxation, time_limit=3)
    optimum = min(energy(h, J, dict(zip(h, state)), .1) for state in product([-1, 1], repeat=len(h)))
    assert result.exact_cut_lower_bound <= optimum == result.exact_energy
    assert result.exact_energy == energy(h, J, result.spins, .1)
    witness = result.certificate.to_dict()
    # Checking the saved witness must work without any numerical solver.
    import highs_turbo.ising_sdp as sdp
    for name in ['eigh', 'minimize', 'cholesky']:
        monkeypatch.setattr(sdp, name, lambda *a, **k: pytest.fail('No numerical solver in checker'))
    assert verify_ising_certificate(h, J, witness, offset=.1)
    if witness['version'] == 2:
        altered = copy.deepcopy(witness)
        altered['gram_factor'][0][0] = 2**53
        assert not verify_ising_certificate(h, J, altered, offset=.1)
        altered = copy.deepcopy(witness)
        altered['gram_denominator'] = 0
        assert not verify_ising_certificate(h, J, altered, offset=.1)
    witness['lower_bound'] = [123, 1]
    assert not verify_ising_certificate(h, J, witness, offset=.1)
