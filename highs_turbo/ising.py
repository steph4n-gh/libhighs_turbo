"""Sparse Ising optimization with valid cycle cuts and explicit optimality gaps."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
import time

import highspy
import networkx as nx
import numpy as np
import scipy.sparse as sp

from highs_turbo.graph_generator import GraphInstance
from highs_turbo.lp_accelerator import _check_status, _add_rows
from highs_turbo.rational_verifier import RationalCutVerifier, VerificationCertificate


@dataclass
class IsingResult:
    """Bounds refer to the original energy, including its constant offset.

    OPTIMAL is HiGHS' numerical conclusion. Exact rational certification is
    reported separately, only when a verified cut bound equals the spin energy.
    """

    spins: dict
    energy: float
    lower_bound: float
    gap: float
    relative_gap: float
    status: str
    message: str
    solve_time: float
    node_count: int
    cuts_added: int
    cut_lower_bound: float
    exact_energy: Fraction
    exact_cut_lower_bound: Fraction
    is_rationally_certified: bool
    cut_certificate: VerificationCertificate | None = None

    @property
    def success(self):
        return self.status == "OPTIMAL"


def _downward(value):
    rounded = float(value)
    return float(np.nextafter(rounded, -np.inf)) if Fraction(rounded) > value else rounded


def _normalize(h, J, offset):
    fields = dict(h) if hasattr(h, "items") else dict(enumerate(h))
    labels = dict.fromkeys(fields)
    quadratic = {}
    constant = Fraction.from_float(float(offset))
    for (u, v), value in J.items():
        labels[u] = labels[v] = None
        value = Fraction.from_float(float(value))
        if u == v:
            constant += value
        else:
            # Mixed, hashable node labels need not be mutually sortable.
            key = frozenset((u, v))
            quadratic[key] = quadratic.get(key, Fraction()) + value
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    fields = [Fraction.from_float(float(fields.get(label, 0))) for label in labels]
    couplings = {}
    for pair, value in quadratic.items():
        u, v = sorted(index[label] for label in pair)
        if value:
            couplings[u, v] = value
    return labels, fields, couplings, constant


def _cycle_rows(graph, weights, deadline):
    """Triangle facets, chordless squares, and frustrated fundamental cycles."""
    cycles = set()
    for u in graph:
        for v in graph[u]:
            if v > u:
                for w in graph[u].keys() & graph[v].keys():
                    if w > v:
                        cycles.add((u, v, w))
        if time.perf_counter() >= deadline:
            break
    for u in graph:
        for v, w in combinations(sorted(v for v in graph[u] if v > u), 2):
            if graph.has_edge(v, w):
                continue
            for t in graph[v].keys() & graph[w].keys():
                if t > u and not graph.has_edge(u, t):
                    cycles.add((u, v, t, w))
        if time.perf_counter() >= deadline:
            break
    if time.perf_counter() < deadline:
        for cycle in nx.cycle_basis(graph):
            start = cycle.index(min(cycle))
            cycle = cycle[start:] + cycle[:start]
            if cycle[1] > cycle[-1]:
                cycle = cycle[:1] + cycle[:0:-1]
            cycles.add(tuple(cycle))
    specs = []
    for cycle in sorted(cycles, key=lambda c: (len(c), c)):
        edges = [tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
                 for i in range(len(cycle))]
        if len(cycle) == 3:
            choices = [[edge] for edge in edges] + [edges]
        else:
            preferred = [edge for edge in edges if weights[edge] > 0]
            choices = [preferred] if len(preferred) % 2 else []
        specs.extend((cycle, positive, 1) for positive in choices)
    return specs


def _cut_data(graph, edges, weights, constant, deadline):
    """Verify each retained cut, then pack frustrated cycles into a bound.

    For a verified aggregate a*x <= b and cut variables 0 <= x <= 1,
    w*x <= b + sum(max(0, w-a)). Convert that Max-Cut upper bound back
    to an Ising lower bound using E = constant + sum(w) - 2*w*x.
    All certificate and objective arithmetic here is exact rational arithmetic.
    """
    specs = _cycle_rows(graph, weights, deadline)
    instance = GraphInstance("ising", len(graph), edges)
    verifier = RationalCutVerifier()
    # This verifies every support, edge membership, odd parity, and multiplier.
    verified = verifier.verify_cycle_conic_combination(instance, specs)
    if not verified.is_valid:
        raise RuntimeError("Invalid cycle cut: " + verified.status)
    edge_index = {edge: i for i, edge in enumerate(edges)}
    indices, values, starts, rhs = [], [], [0], []
    remaining = {edge: abs(weights[edge]) for edge in edges}
    packing = []
    for cycle, positive, _ in specs:
        cycle_edges = [tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
                       for i in range(len(cycle))]
        positive = set(positive)
        indices.extend(edge_index[e] for e in cycle_edges)
        values.extend(1 if e in positive else -1 for e in cycle_edges)
        starts.append(len(indices))
        rhs.append(len(positive) - 1)
        if all((weights[e] > 0) == (e in positive) for e in cycle_edges):
            multiplier = min(remaining[e] for e in cycle_edges)
            if multiplier:
                packing.append((cycle, list(positive), multiplier))
                for edge in cycle_edges:
                    remaining[edge] -= multiplier
    certificate = verifier.verify_cycle_conic_combination(instance, packing)
    if not certificate.is_valid:
        raise RuntimeError("Invalid surrogate combination: " + certificate.status)
    upper_cut = certificate.exact_rhs + sum(
        (max(Fraction(), weights[e] - certificate.exact_coefficients.get(e, 0))
         for e in edges), Fraction())
    lower = constant + sum(weights.values(), Fraction()) - 2 * upper_cut
    matrix = sp.csr_matrix((values, indices, starts), shape=(len(rhs), len(edges)), dtype=float)
    return matrix, np.asarray(rhs, dtype=float), lower, certificate


def _initial_spins(n, edges, weights, seed, deadline):
    """A bounded multistart descent supplies a feasible incumbent to HiGHS."""
    u, v = np.asarray(edges, dtype=int).T
    w = np.asarray([float(weights[e]) for e in edges])
    adjacency = sp.csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    rng = np.random.default_rng(seed)
    best = np.ones(n, dtype=np.int8)
    best_value = float(w.sum())
    for attempt in range(8):
        spins = np.ones(n, dtype=np.int8) if attempt == 0 else rng.choice([-1, 1], size=n)
        fields = adjacency @ spins
        for _ in range(8 * n):
            gain = spins * fields
            vertex = int(np.argmax(gain))
            if gain[vertex] <= 1e-12 or time.perf_counter() >= deadline:
                break
            start, end = adjacency.indptr[vertex:vertex + 2]
            fields[adjacency.indices[start:end]] -= 2 * spins[vertex] * adjacency.data[start:end]
            spins[vertex] *= -1
        value = float(w @ (spins[u] * spins[v]))
        if value < best_value:
            best, best_value = spins.copy(), value
        if time.perf_counter() >= deadline:
            break
    return best


def solve_ising(h, J, *, offset=0.0, time_limit=None, relative_gap=0.0,
                seed=0, accelerate=True):
    """Minimize offset + sum(h[i]*s[i]) + sum(J[i,j]*s[i]*s[j]), s in {-1,1}.

    h is a mapping or sequence; J maps pairs of arbitrary hashable labels to
    couplings. Both orientations add, self-couplings add to the constant.
    time_limit includes preparation; preprocessing may exceed very tiny limits.
    accelerate=False uses the same sparse HiGHS model without cuts or a start.
    No QPU, compiled graph extension, or neural weights are required.
    """
    started = time.perf_counter()
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative, or None")
    if not math.isfinite(relative_gap) or relative_gap < 0:
        raise ValueError("relative_gap must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    deadline = started + (time_limit if time_limit is not None else math.inf)
    try:
        labels, fields, couplings, constant = _normalize(h, J, offset)
    except (OverflowError, ValueError) as error:
        raise ValueError("Ising coefficients must be finite numbers") from error
    n = len(labels)
    # A fixed reference spin turns local fields into ordinary graph edges.
    weights = dict(couplings)
    weights.update({(i, n): value for i, value in enumerate(fields) if value})
    edges = sorted(weights)
    vertices = n + bool(any(fields))
    graph = nx.Graph()
    graph.add_nodes_from(range(vertices))
    graph.add_edges_from(edges)
    spins = np.ones(vertices, dtype=int)
    trivial = constant - sum((abs(w) for w in weights.values()), Fraction())
    exact_lower, certificate = trivial, None
    matrix = sp.csr_matrix((0, len(edges)))
    rhs = np.empty(0)
    prep_deadline = min(deadline, started + (0.1 * time_limit if time_limit else 1.0))
    if edges and accelerate and time.perf_counter() < prep_deadline:
        # Reserve a short first pass for the incumbent: on full fabrics, cut
        # verification can otherwise consume the entire preparation budget.
        spins = _initial_spins(vertices, edges, weights, seed,
                               min(prep_deadline, time.perf_counter() + 0.1))
        matrix, rhs, exact_lower, certificate = _cut_data(graph, edges, weights, constant, prep_deadline)
    # Fix one orientation per connected component (the reference spin when present).
    components = list(nx.connected_components(graph))
    anchors = [n if n in component else min(component) for component in components]
    for component, anchor in zip(components, anchors):
        if spins[anchor] < 0:
            spins[list(component)] *= -1

    def exact_energy(state):
        return constant + sum((w * int(state[u] * state[v]) for (u, v), w in weights.items()), Fraction())

    energy = exact_energy(spins)
    numerical_lower = _downward(exact_lower)
    status, message, nodes, cuts_added = "TIME_LIMIT", "Time limit reached during preparation", 0, 0
    if energy == exact_lower:
        status, message = "OPTIMAL", "Spin energy equals the exact rational cut bound"
    elif time.perf_counter() < deadline:
        m = len(edges)
        u, v = np.asarray(edges, dtype=int).T
        costs = np.r_[-2 * np.asarray([float(weights[e]) for e in edges]), np.zeros(vertices)]
        lower, upper = np.zeros(m + vertices), np.ones(m + vertices)
        upper[m + np.asarray(anchors)] = 0
        rows = np.repeat(np.arange(4 * m), 3)
        cols = np.stack([np.repeat(np.arange(m), 4),
                         np.repeat(m + u, 4), np.repeat(m + v, 4)], axis=1).ravel()
        data = np.tile([1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1], m)
        constraints = sp.csr_matrix((data, (rows, cols)), shape=(4 * m, m + vertices))
        bounds = np.tile([0, 2, 0, 0], m)
        session = highspy.Highs()
        for key, value in {"output_flag": False, "parallel": "off", "random_seed": int(seed),
                           "mip_rel_gap": float(relative_gap), "mip_abs_gap": 0.0}.items():
            _check_status(session.setOptionValue(key, value))
        _check_status(session.addCols(m + vertices, costs, lower, upper, 0, [], [], []))
        _check_status(session.changeColsIntegrality(vertices, np.arange(m, m + vertices), np.ones(vertices, dtype=np.uint8)))
        _add_rows(session, constraints, np.full(4 * m, -np.inf), bounds)
        if matrix.shape[0]:
            _add_rows(session, sp.hstack([matrix, sp.csr_matrix((len(rhs), vertices))]).tocsr(),
                      np.full(len(rhs), -np.inf), rhs)
            cuts_added = len(rhs)
        objective_constant = constant + sum(weights.values(), Fraction())
        _check_status(session.changeObjectiveOffset(float(objective_constant)))
        if accelerate:
            values = np.r_[(spins[u] != spins[v]).astype(float), (1 - spins) / 2]
            _check_status(session.setSolution(len(values), np.arange(len(values)), values))
        _check_status(session.setOptionValue("time_limit", max(0.0, deadline - time.perf_counter())))
        _check_status(session.run())
        info, solution = session.getInfo(), session.getSolution()
        if solution.value_valid:
            candidate = 1 - 2 * np.rint(solution.col_value[m:]).astype(int)
            if np.isin(candidate, [-1, 1]).all():
                candidate_energy = exact_energy(candidate)
                if candidate_energy < energy:
                    spins, energy = candidate, candidate_energy
        if info.valid and math.isfinite(info.mip_dual_bound):
            numerical_lower = max(numerical_lower, min(float(energy), info.mip_dual_bound))
        model_status = session.getModelStatus()
        status = {highspy.HighsModelStatus.kOptimal: "OPTIMAL",
                  highspy.HighsModelStatus.kTimeLimit: "TIME_LIMIT",
                  highspy.HighsModelStatus.kInterrupt: "INTERRUPTED"}.get(model_status, "SOLVER_ERROR")
        message = session.modelStatusToString(model_status)
        nodes = max(0, info.mip_node_count)
    exact_optimal = energy == exact_lower
    if exact_optimal:
        status, message = "OPTIMAL", "Spin energy equals the exact rational cut bound"
    gap = 0.0 if exact_optimal else max(0.0, float(energy) - numerical_lower)
    if status == "OPTIMAL" and gap > max(1e-6, abs(float(energy)) * 1e-9):
        status = "GAP_LIMIT"
    return IsingResult(
        spins={label: int(spins[i]) for i, label in enumerate(labels)}, energy=float(energy),
        lower_bound=numerical_lower, gap=gap, relative_gap=gap / max(abs(float(energy)), 1e-10),
        status=status, message=message, solve_time=time.perf_counter() - started,
        node_count=nodes, cuts_added=cuts_added, cut_lower_bound=_downward(exact_lower),
        exact_energy=energy, exact_cut_lower_bound=exact_lower,
        is_rationally_certified=exact_optimal, cut_certificate=certificate,
    )
