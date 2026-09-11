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
from highs_turbo.ising_cuts import (
    IsingCertificate, check_certificate, cut_matrix, cycle_cut, make_certificate,
    optimize_dual, separate_cycles, separate_short_cycles, square_indices, subgraph_cut, triangle_indices, verify_cut,
)
from highs_turbo.ising_policy import cluster_candidates, rank_clusters


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
    certificate: IsingCertificate | None = None
    exact_gap: Fraction = Fraction()
    root_rounds: int = 0
    subgraph_cuts: int = 0
    progress: tuple = ()

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


def _cut_data(graph, edges, weights, constant, deadline, *, with_proof=False):
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
    proof_cuts, proof_duals = [], []
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
                proof_cuts.append(cycle_cut(cycle, positive, edge_index))
                proof_duals.append(-multiplier)
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
    result = matrix, np.asarray(rhs, dtype=float), lower, certificate
    return (*result, make_certificate(proof_cuts, proof_duals, edges, weights, constant)) if with_proof else result


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


def _sample_spins(n, edges, weights, seed, deadline):
    """Use a compiled CPU sampler, retaining descent as an optional-dependency fallback."""
    try:
        from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver, TabuSampler
    except ImportError:
        return _initial_spins(n, edges, weights, seed, deadline)
    h = {i: 0. for i in range(n)}
    J = {edge: float(weights[edge]) for edge in edges}
    annealing_deadline = time.perf_counter()+max(0., deadline-time.perf_counter())*.6
    samples = SimulatedAnnealingSampler().sample_ising(
        h, J, num_reads=16, num_sweeps=400, seed=int(seed),
        interrupt_function=lambda: time.perf_counter() >= annealing_deadline,
    )
    remaining_ms = int(1000*(deadline-time.perf_counter()))
    if remaining_ms > 5:
        improved = TabuSampler().sample_ising(
            h, J, initial_states=samples.first.sample, num_reads=1,
            timeout=min(remaining_ms, 30), seed=int(seed))
        if improved.first.energy < samples.first.energy:
            samples = improved
    if time.perf_counter() < deadline:
        samples = SteepestDescentSolver().sample_ising(h, J, initial_states=samples)
    return np.asarray([samples.first.sample[i] for i in range(n)], dtype=int)


def _adaptive_relaxation(graph, edges, weights, constant, energy, *, deadline, started,
                         seed, policy, threads, certified_gap=None, training_samples=None):
    """Keep one LP and its basis while adding verified, solution-dependent cuts."""
    m = len(edges)
    edge_index = {e: i for i, e in enumerate(edges)}
    cuts, seen = [], set()
    session = highspy.Highs()
    for key, value in {"output_flag": False, "parallel": "off", "threads": threads,
                       "solver": "simplex", "presolve": "off", "random_seed": int(seed)}.items():
        _check_status(session.setOptionValue(key, value))
    objective = np.asarray([float(weights[e]) for e in edges])
    _check_status(session.addCols(m, -objective, np.zeros(m), np.ones(m), 0, [], [], []))

    def add(candidates):
        accepted = []
        for cut in candidates:
            key = (cut.indices, cut.coefficients, cut.rhs)
            if key in seen:
                continue
            if not verify_cut(cut, edges):
                raise RuntimeError("Adaptive separator proposed an invalid cut")
            seen.add(key)
            accepted.append(cut)
        if accepted:
            _add_rows(session, cut_matrix(accepted, m), np.full(len(accepted), -np.inf),
                      np.asarray([cut.rhs for cut in accepted], dtype=float))
            cuts.extend(accepted)
        return len(accepted)

    triangles = triangle_indices(graph, edges)
    squares = square_indices(graph, edges)
    preferred = (objective > 0).astype(float)
    initial = separate_short_cycles(triangles, preferred)+separate_short_cycles(squares, preferred)
    # Coordinate search cheaply combines the whole pool before simplex.
    # Packing supplies an exact feasible starting multiplier vector.
    remaining = [abs(weights[e]) for e in edges]
    packed, packing = [], []
    for cut in initial:
        value = min(remaining[i] for i in cut.indices)
        packed.append(cut)
        packing.append(-value)
        for i in cut.indices:
            remaining[i] -= value
    add(packed)
    best = make_certificate(cuts, packing, edges, weights, constant)
    progress, rounds, no_gain = [], 0, 0

    def record_progress():
        if not check_certificate(best, edges, weights, constant):
            raise RuntimeError("Ising progress certificate failed verification")
        progress.append({"time": time.perf_counter()-started, "lower_bound": _downward(best.lower_bound),
                         "energy": float(energy), "cuts": len(cuts), "round": rounds})

    if time.perf_counter() < deadline and training_samples is None:
        dual, point = optimize_dual(cuts, objective, -np.asarray(packing, dtype=float),
                               min(deadline, time.perf_counter()+0.25), seed)
        proof = make_certificate(cuts, -dual, edges, weights, constant)
        if proof.lower_bound > best.lower_bound:
            best = proof
        record_progress()
        # The averaged dual residual signs provide an inexpensive fractional
        # point for separation before the first, potentially costly LP solve.
        # It need not be feasible: every proposed inequality is checked.
        fast_deadline = min(deadline, time.perf_counter()+0.5)
        for iteration in range(2):
            if (time.perf_counter() >= fast_deadline or energy == best.lower_bound
                    or (certified_gap is not None and energy-best.lower_bound <= certified_gap)):
                break
            u, v = np.asarray(edges).T
            fractional = np.minimum(point, 1-point)
            scores = np.bincount(np.r_[u, v], weights=np.tile(fractional, 2), minlength=len(graph))
            roots = np.argsort(-scores, kind="stable")[:32]
            supports, features = cluster_candidates(graph, edges, weights, point, roots, fast_deadline)
            proposed = (separate_short_cycles(triangles, point, limit=256)
                        + separate_short_cycles(squares, point, limit=256))
            for index in rank_clusters(features, policy)[:8]:
                if time.perf_counter() >= fast_deadline:
                    break
                cut = subgraph_cut(supports[index], edges, weights, point,
                                   min(fast_deadline, time.perf_counter()+0.025), threads)
                if cut is not None:
                    proposed.append(cut)
            if not add(proposed):
                break
            dual = np.pad(dual, (0, len(cuts)-len(dual)))
            dual, point = optimize_dual(cuts, objective, dual, fast_deadline, seed+iteration+1)
            proof = make_certificate(cuts, -dual, edges, weights, constant)
            if proof.lower_bound > best.lower_bound:
                best = proof
            record_progress()
    last_objective, cluster_points = None, {}
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(graph))
    while time.perf_counter() < deadline:
        if energy == best.lower_bound or (certified_gap is not None and energy-best.lower_bound <= certified_gap):
            break
        _check_status(session.setOptionValue("time_limit", max(0., deadline-time.perf_counter())))
        _check_status(session.run())
        solution = session.getSolution()
        if not solution.value_valid:
            break
        x = np.asarray(solution.col_value)
        if len(solution.row_dual) == len(cuts):
            proof = make_certificate(cuts, solution.row_dual, edges, weights, constant)
            if proof.lower_bound > best.lower_bound:
                best = proof
        rounds += 1
        record_progress()
        if energy == best.lower_bound or (certified_gap is not None and energy-best.lower_bound <= certified_gap):
            break
        if session.getModelStatus() != highspy.HighsModelStatus.kOptimal or time.perf_counter() >= deadline:
            break
        current_objective = session.getObjectiveValue()
        no_gain = no_gain+1 if last_objective is not None and abs(current_objective-last_objective) < 1e-8 else 0
        last_objective = current_objective
        fractionality = np.minimum(np.clip(x, 0, 1), 1-np.clip(x, 0, 1))
        u, v = np.asarray(edges).T
        node_scores = np.bincount(np.r_[u, v], weights=np.tile(fractionality, 2), minlength=len(graph))
        ranked = np.argsort(-node_scores, kind="stable")
        offset = ((rounds-1)*64) % len(graph)
        roots = list(dict.fromkeys(np.r_[ranked[:64], np.roll(shuffled, -offset)[:64]].tolist()))
        cycle_candidates = (separate_short_cycles(triangles, x, limit=1024)
                            + separate_short_cycles(squares, x, limit=1024))
        if len(cycle_candidates) < 256:
            cycle_candidates += separate_cycles(graph, edges, x, roots, deadline)
        # Explore larger supports each round, including while cycle cuts are
        # still available; otherwise they can be starved on large fabrics.
        candidates, features = cluster_candidates(graph, edges, weights, x, roots[:32], deadline)
        order = rank_clusters(features, policy) if training_samples is None else rng.permutation(len(candidates))
        cluster_cuts = []
        count = len(order) if training_samples is not None else 8
        attempted = 0
        for index in order:
            if time.perf_counter() >= deadline:
                break
            point_key = np.round(x[list(candidates[index])], 7).tobytes()
            if cluster_points.get(candidates[index]) == point_key:
                continue
            cluster_points[candidates[index]] = point_key
            attempted += 1
            before = time.perf_counter()
            cut = subgraph_cut(candidates[index], edges, weights, x,
                               min(deadline, before+0.05), threads)
            elapsed = time.perf_counter()-before
            violation, bound_gain = 0., None
            if cut is not None:
                violation = (sum(a*x[i] for i, a in zip(cut.indices, cut.coefficients))-cut.rhs)
                violation /= max(1, np.linalg.norm(cut.coefficients))
                cluster_cuts.append(cut)
                if training_samples is not None and time.perf_counter() < deadline:
                    # Measure the candidate's actual marginal root-bound
                    # improvement, then restore the same model and basis.
                    if not verify_cut(cut, edges):
                        raise RuntimeError("Invalid training candidate")
                    basis = session.getBasis()
                    _add_rows(session, cut_matrix([cut], m), np.asarray([-np.inf]), np.asarray([cut.rhs]))
                    _check_status(session.setOptionValue("time_limit", max(0., min(0.05, deadline-time.perf_counter()))))
                    _check_status(session.run())
                    if session.getModelStatus() == highspy.HighsModelStatus.kOptimal:
                        bound_gain = max(0., 2*(session.getObjectiveValue()-current_objective))
                    _check_status(session.deleteRows(1, np.asarray([len(cuts)], dtype=np.int32)))
                    _check_status(session.setBasis(basis))
            else:
                bound_gain = 0.
            if training_samples is not None:
                training_samples.append({"features": features[index].tolist(), "violation": violation,
                                         "seconds": elapsed, "bound_gain": bound_gain, "round": rounds})
            if attempted >= count:
                break
        added = add(cycle_candidates+cluster_cuts)
        if not added or no_gain >= 4:
            break
    # No solver state is required to check this witness later.
    return cuts, best, rounds, tuple(progress)


def solve_ising(h, J, *, offset=0.0, time_limit=None, relative_gap=0.0,
                seed=0, accelerate=True, cut_policy="deterministic",
                certified_gap=None, threads=0):
    """Minimize offset + sum(h[i]*s[i]) + sum(J[i,j]*s[i]*s[j]), s in {-1,1}.

    h is a mapping or sequence; J maps pairs of arbitrary hashable labels to
    couplings. Both orientations add, self-couplings add to the constant.
    time_limit includes preparation; preprocessing may exceed very tiny limits.
    accelerate=False uses the same sparse HiGHS model without cuts or a start.
    cut_policy selects deterministic or learned adaptive separation; "static"
    retains the original one-pass acceleration for comparison. certified_gap
    is an optional absolute target checked against the rational certificate.
    threads=0 leaves the native runtime's thread count automatic.
    """
    started = time.perf_counter()
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative, or None")
    if not math.isfinite(relative_gap) or relative_gap < 0:
        raise ValueError("relative_gap must be finite and nonnegative")
    if certified_gap is not None and (not math.isfinite(certified_gap) or certified_gap < 0):
        raise ValueError("certified_gap must be finite and nonnegative, or None")
    if isinstance(threads, bool) or not isinstance(threads, (int, np.integer)) or threads < 0:
        raise ValueError("threads must be a nonnegative integer")
    if cut_policy not in ("deterministic", "learned", "static"):
        raise ValueError("cut_policy must be 'deterministic', 'learned', or 'static'")
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
    proof = make_certificate([], [], edges, weights, constant)
    root_rounds, subgraph_count, progress = 0, 0, ()
    matrix = sp.csr_matrix((0, len(edges)))
    rhs = np.empty(0)
    prep_deadline = min(deadline, started + (0.1 * time_limit if time_limit else 1.0))
    if edges and accelerate and time.perf_counter() < prep_deadline:
        initializer = _initial_spins if cut_policy == "static" else _sample_spins
        spins = initializer(vertices, edges, weights, seed,
                            min(prep_deadline, time.perf_counter() + 0.1))
        if cut_policy == "static":
            matrix, rhs, exact_lower, certificate, proof = _cut_data(
                graph, edges, weights, constant, prep_deadline, with_proof=True)
            exact_lower = proof.lower_bound
    # Fix one orientation per connected component (the reference spin when present).
    components = list(nx.connected_components(graph))
    anchors = [n if n in component else min(component) for component in components]
    for component, anchor in zip(components, anchors):
        if spins[anchor] < 0:
            spins[list(component)] *= -1

    def exact_energy(state):
        return constant + sum((w * int(state[u] * state[v]) for (u, v), w in weights.items()), Fraction())

    energy = exact_energy(spins)
    if edges and accelerate and cut_policy != "static" and time.perf_counter() < deadline:
        root_deadline = deadline if certified_gap is not None else min(
            deadline, started + (0.7*time_limit if time_limit else 3.0))
        cuts, proof, root_rounds, progress = _adaptive_relaxation(
            graph, edges, weights, constant, energy, deadline=root_deadline, started=started,
            seed=seed, policy=cut_policy, threads=int(threads),
            certified_gap=None if certified_gap is None else Fraction(float(certified_gap)),
        )
        exact_lower = proof.lower_bound
        matrix = cut_matrix(cuts, len(edges))
        rhs = np.asarray([cut.rhs for cut in cuts], dtype=float)
        subgraph_count = sum(c.kind == "subgraph" for c in cuts)
    numerical_lower = _downward(exact_lower)
    status, message, nodes, cuts_added = "TIME_LIMIT", "Time limit reached during preparation", 0, len(rhs)
    if energy == exact_lower:
        status, message = "OPTIMAL", "Spin energy equals the exact rational cut bound"
    elif certified_gap is not None and energy-proof.lower_bound <= Fraction(float(certified_gap)):
        status, message = "GAP_LIMIT", "Requested independently verified gap reached"
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
        for key, value in {"output_flag": False, "parallel": "off", "threads": int(threads), "random_seed": int(seed),
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
    if not check_certificate(proof, edges, weights, constant):
        raise RuntimeError("Final Ising certificate failed independent verification")
    if energy < proof.lower_bound:
        raise RuntimeError("Verified bound exceeds the independently evaluated spin energy")
    progress = (*progress, {"time": time.perf_counter()-started, "lower_bound": _downward(proof.lower_bound),
                           "energy": float(energy), "cuts": matrix.shape[0], "round": root_rounds})
    return IsingResult(
        spins={label: int(spins[i]) for i, label in enumerate(labels)}, energy=float(energy),
        lower_bound=numerical_lower, gap=gap, relative_gap=gap / max(abs(float(energy)), 1e-10),
        status=status, message=message, solve_time=time.perf_counter() - started,
        node_count=nodes, cuts_added=cuts_added, cut_lower_bound=_downward(exact_lower),
        exact_energy=energy, exact_cut_lower_bound=exact_lower,
        is_rationally_certified=exact_optimal, cut_certificate=certificate,
        certificate=proof, exact_gap=energy-proof.lower_bound, root_rounds=root_rounds,
        subgraph_cuts=subgraph_count, progress=progress,
    )


def verify_ising_certificate(h, J, certificate, *, offset=0.0):
    """Check a serialized or in-memory bound witness without running any solver."""
    try:
        if isinstance(certificate, dict):
            certificate = IsingCertificate.from_dict(certificate)
        labels, fields, weights, constant = _normalize(h, J, offset)
        weights.update({(i, len(labels)): value for i, value in enumerate(fields) if value})
        return check_certificate(certificate, sorted(weights), weights, constant)
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False
