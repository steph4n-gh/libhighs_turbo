"""Adaptive cut separation and independently checkable Ising lower bounds.

The search routines may use floating point. A cut enters the LP only after
its integer inequality has been checked, and a bound uses nonnegative dyadic
multipliers with an exact residual correction. No LP optimality assumption
is needed by the certificate checker.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from hashlib import sha256
from itertools import combinations
from math import gcd, lcm
import json
import time

import highspy
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from highs_turbo.lp_accelerator import _add_rows, _check_status


MAX_SUBGRAPH_VERTICES = 12


@dataclass(frozen=True)
class IsingCut:
    indices: tuple[int, ...]
    coefficients: tuple[int, ...]
    rhs: int
    kind: str


@dataclass(frozen=True)
class IsingCertificate:
    """A bound witness in the normalized problem's edge coordinates.

    Multipliers are integer numerators over one positive denominator. The
    digest binds the normalized coefficients, including the energy offset.
    It is an identity check; validity comes from checking every inequality
    and recomputing the rational bound.
    """

    problem_digest: str
    cuts: tuple[IsingCut, ...]
    multipliers: tuple[int, ...]
    denominator: int
    lower_bound: Fraction
    gram_factor: tuple[tuple[int, ...], ...] = ()
    gram_denominator: int = 1
    extra_edges: tuple[tuple[int, int], ...] = ()
    sparse_gram_factor: tuple[tuple[int, ...], ...] = ()

    def to_dict(self):
        data = {
            "version": 4 if self.sparse_gram_factor else 3 if self.extra_edges else 2 if self.gram_factor else 1,
            "problem_digest": self.problem_digest,
            "cuts": [{"indices": list(c.indices), "coefficients": list(c.coefficients),
                      "rhs": c.rhs, "kind": c.kind} for c in self.cuts],
            "multipliers": list(self.multipliers), "denominator": self.denominator,
            "lower_bound": [self.lower_bound.numerator, self.lower_bound.denominator],
        }
        if self.gram_factor or self.extra_edges or self.sparse_gram_factor:
            data.update(gram_factor=[list(row) for row in self.gram_factor],
                        gram_denominator=self.gram_denominator)
        if self.extra_edges:
            data["extra_edges"] = [list(edge) for edge in self.extra_edges]
        if self.sparse_gram_factor:
            data["sparse_gram_factor"] = [list(row) for row in self.sparse_gram_factor]
        return data

    @classmethod
    def from_dict(cls, data):
        if data["version"] not in (1, 2, 3, 4):
            raise ValueError("Unsupported Ising certificate version")
        return cls(data["problem_digest"],
                   tuple(IsingCut(tuple(c["indices"]), tuple(c["coefficients"]), c["rhs"], c["kind"])
                         for c in data["cuts"]),
                   tuple(data["multipliers"]), data["denominator"], Fraction(*data["lower_bound"]),
                   tuple(tuple(row) for row in data["gram_factor"]) if data["version"] >= 2 else (),
                   data["gram_denominator"] if data["version"] >= 2 else 1,
                   tuple(tuple(edge) for edge in data.get("extra_edges", ())) if data["version"] >= 3 else (),
                   tuple(tuple(row) for row in data["sparse_gram_factor"]) if data["version"] == 4 else ())


def problem_digest(edges, weights, constant):
    data = [[u, v, weights[u, v].numerator, weights[u, v].denominator] for u, v in edges]
    data.append([constant.numerator, constant.denominator])
    return sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def cut_assignments(edges):
    """Enumerate all cuts of a small support, fixing one global orientation."""
    vertices = sorted({v for edge in edges for v in edge})
    if len(vertices) > MAX_SUBGRAPH_VERTICES:
        raise ValueError("Subgraph exceeds the exhaustive verification limit")
    index = {v: i for i, v in enumerate(vertices)}
    states = (np.arange(1 << max(0, len(vertices) - 1), dtype=np.uint32) << 1)[:, None]
    u = np.asarray([index[u] for u, _ in edges], dtype=np.uint32)
    v = np.asarray([index[v] for _, v in edges], dtype=np.uint32)
    return (((states >> u) ^ (states >> v)) & 1).astype(np.int8)


def verify_cut(cut, edges):
    """Check topology/parity or exhaustively check a bounded integer support."""
    if (not isinstance(cut, IsingCut) or not cut.indices
            or len(cut.indices) != len(cut.coefficients)
            or any(type(i) is not int or not 0 <= i < len(edges) for i in cut.indices)
            or tuple(sorted(set(cut.indices))) != cut.indices
            or any(type(a) is not int or a == 0 for a in cut.coefficients)
            or type(cut.rhs) is not int
            or sum(abs(a) for a in cut.coefficients) >= 2**52):
        return False
    support = [edges[i] for i in cut.indices]
    if cut.kind == "cycle":
        positive = sum(a == 1 for a in cut.coefficients)
        if (len(support) < 3 or any(abs(a) != 1 for a in cut.coefficients)
                or positive % 2 != 1 or cut.rhs != positive - 1):
            return False
        adjacency = {}
        for u, v in support:
            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return False
        reached, stack = set(), [support[0][0]]
        while stack:
            vertex = stack.pop()
            if vertex not in reached:
                reached.add(vertex)
                stack.extend(adjacency[vertex])
        return len(reached) == len(adjacency)
    if cut.kind == "subgraph":
        if len({v for edge in support for v in edge}) > MAX_SUBGRAPH_VERTICES:
            return False
        # Each dot product is an exact int64 sum within the checked range.
        values = cut_assignments(support) @ np.asarray(cut.coefficients, dtype=np.int64)
        return int(values.max()) <= cut.rhs
    return False


def cut_matrix(cuts, num_edges):
    starts = [0]
    indices, values = [], []
    for cut in cuts:
        indices.extend(cut.indices)
        values.extend(cut.coefficients)
        starts.append(len(indices))
    return sp.csr_matrix((values, indices, starts), shape=(len(cuts), num_edges), dtype=float)


def optimize_dual(cuts, weights, multipliers, deadline, seed, sweeps=64):
    """Search for useful multipliers; validity is established by the checker."""
    matrix = cut_matrix(cuts, len(weights))
    rhs = np.asarray([c.rhs for c in cuts], dtype=float)
    multipliers = np.asarray(multipliers, dtype=float)
    remaining = max(0., deadline-time.perf_counter())
    try:
        from highs_turbo._compiled_engine import ising_dual_ascent
    except ImportError:
        ising_dual_ascent = None
    if ising_dual_ascent is not None:
        dual, point = ising_dual_ascent(matrix.indptr, matrix.indices, matrix.data,
                                       rhs, weights, multipliers, sweeps, remaining, int(seed))
        return np.asarray(dual), np.asarray(point)
    # Portable implementation of the same coordinate rule. Its loop checks
    # the deadline within each sweep, because Python iteration is slower.
    residual = np.asarray(weights)-matrix.T @ multipliers
    point = np.where(residual > 1e-8, 1., np.where(residual < -1e-8, 0., .5))
    best, best_value = multipliers.copy(), float(rhs @ multipliers + np.maximum(0, residual).sum())
    rng = np.random.default_rng(seed)
    for sweep in range(sweeps):
        for row in rng.permutation(len(cuts)):
            if time.perf_counter() >= deadline:
                return best, point
            cut = cuts[row]
            target = sum(max(0, a) for a in cut.coefficients)-cut.rhs
            crossings = sorted((multipliers[row]+residual[i]/a, abs(a))
                               for i, a in zip(cut.indices, cut.coefficients))
            next_value, accumulated = 0., 0
            if target > 0:
                for position, (value, jump) in enumerate(crossings):
                    accumulated += jump
                    if accumulated >= target:
                        next_value = max(0, value)
                        if accumulated == target and position+1 < len(crossings):
                            next_value = (next_value+max(0, crossings[position+1][0]))/2
                        break
            delta = next_value-multipliers[row]
            for i, a in zip(cut.indices, cut.coefficients):
                residual[i] -= delta*a
            multipliers[row] = next_value
        current = np.where(residual > 1e-8, 1., np.where(residual < -1e-8, 0., .5))
        point = current if sweep == 0 else .8*point+.2*current
        value = float(rhs @ multipliers + np.maximum(0, residual).sum())
        if value < best_value:
            best, best_value = multipliers.copy(), value
    return best, point


def _bound(cuts, multipliers, denominator, edges, weights, constant,
           gram_factor=(), gram_denominator=1, sparse_gram_factor=()):
    """Exact nonnegative aggregation, box residual, and objective-lattice bound."""
    scale = lcm(denominator, constant.denominator, *(weights[e].denominator for e in edges))
    integer_weights = [weights[e].numerator * (scale // weights[e].denominator) for e in edges]
    aggregate = [0] * len(edges)
    rhs = 0
    for cut, numerator in zip(cuts, multipliers):
        factor = numerator * (scale // denominator)
        rhs += factor * cut.rhs
        for index, coefficient in zip(cut.indices, cut.coefficients):
            aggregate[index] += factor * coefficient
    upper_cut = rhs + sum(max(0, w - a) for w, a in zip(integer_weights, aggregate))
    lattice = reduce(gcd, integer_weights, 0)
    if lattice:
        upper_cut = (upper_cut // lattice) * lattice
    lower = constant + Fraction(sum(integer_weights) - 2 * upper_cut, scale)
    if gram_factor or sparse_gram_factor:
        if gram_factor and sparse_gram_factor:
            raise ValueError("A certificate must use one Gram factor format")
        if sparse_gram_factor:
            from highs_turbo.ising_sparse import gram_lower_bound
        else:
            from highs_turbo.ising_sdp import gram_lower_bound
        residual = [w-a for w, a in zip(integer_weights, aggregate)]
        spectral = (constant + Fraction(sum(aggregate)-2*rhs, scale)
                    + gram_lower_bound(edges, residual, scale,
                                       sparse_gram_factor or gram_factor, gram_denominator))
        if lattice:
            # Every original energy is base + an integer multiple of step.
            base = constant + Fraction(sum(integer_weights), scale)
            step = Fraction(2*lattice, scale)
            distance = (spectral-base)/step
            spectral = base + (-(-distance.numerator // distance.denominator))*step
        lower = max(lower, spectral)
    return lower


def make_certificate(cuts, dual, edges, weights, constant):
    # Preserve exact multipliers supplied by combinatorial packing. Numerical
    # LP multipliers are deliberately rounded to a common dyadic grid.
    denominator = lcm(2**20, *(value.denominator for value in dual if isinstance(value, Fraction)))
    active, multipliers = [], []
    for cut, value in zip(cuts, dual):
        if isinstance(value, Fraction):
            numerator = max(0, -value.numerator * (denominator // value.denominator))
        elif not np.isfinite(value):
            continue
        else:
            numerator = max(0, round(-float(value) * 2**20)) * (denominator // 2**20)
        if numerator:
            active.append(cut)
            multipliers.append(numerator)
    lower = _bound(active, multipliers, denominator, edges, weights, constant)
    return IsingCertificate(problem_digest(edges, weights, constant), tuple(active),
                            tuple(multipliers), denominator, lower)


def check_certificate(certificate, edges, weights, constant):
    """Independent of the optimizer: reject bad rows, multipliers, or bounds."""
    if (not isinstance(certificate, IsingCertificate)
            or certificate.problem_digest != problem_digest(edges, weights, constant)
            or type(certificate.denominator) is not int or certificate.denominator <= 0
            or len(certificate.cuts) != len(certificate.multipliers)
            or any(type(x) is not int or x < 0 for x in certificate.multipliers)):
        return False
    try:
        # Auxiliary correlations have zero objective coefficient. Their
        # endpoints must be existing spins, and every new edge occurs once.
        # The digest continues to bind only the original supplied problem.
        n = max((max(edge) for edge in edges), default=-1)+1
        extra = certificate.extra_edges
        if (not isinstance(extra, tuple)
                or any(not isinstance(edge, tuple) or len(edge) != 2
                       or any(type(v) is not int for v in edge)
                       or not 0 <= edge[0] < edge[1] < n for edge in extra)
                or len(set(extra)) != len(extra) or set(extra).intersection(edges)):
            return False
        edges = [*edges, *extra]
        weights = {**weights, **dict.fromkeys(extra, Fraction())}
        if any(not verify_cut(cut, edges) for cut in certificate.cuts):
            return False
        return certificate.lower_bound == _bound(
            certificate.cuts, certificate.multipliers, certificate.denominator, edges, weights, constant,
            certificate.gram_factor, certificate.gram_denominator, certificate.sparse_gram_factor)
    except (TypeError, ValueError, OverflowError):
        return False


def cycle_cut(cycle, positive, edge_index):
    support = [tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)]))) for i in range(len(cycle))]
    positive = set(positive)
    coefficients = {edge_index[e]: 1 if e in positive else -1 for e in support}
    indices = tuple(sorted(coefficients))
    return IsingCut(indices, tuple(coefficients[i] for i in indices), len(positive) - 1, "cycle")


def triangle_indices(graph, edges, deadline=float('inf'), limit=float('inf')):
    edge_index = {e: i for i, e in enumerate(edges)}
    neighbors = [set(graph[u]) for u in range(len(graph))]
    triangles = []
    for u in range(len(graph)):
        above = {v for v in neighbors[u] if v > u}
        for v in above:
            if time.perf_counter() >= deadline or len(triangles) >= limit:
                return np.asarray(triangles, dtype=int).reshape((-1, 3))
            for w in above & neighbors[v]:
                if w > v:
                    triangles.append(sorted((edge_index[u, v], edge_index[u, w], edge_index[v, w])))
    return np.asarray(triangles, dtype=int).reshape((-1, 3))


def square_indices(graph, edges, deadline=float('inf'), limit=float('inf')):
    edge_index = {e: i for i, e in enumerate(edges)}
    neighbors = [set(graph[u]) for u in range(len(graph))]
    squares = []
    for u in range(len(graph)):
        above = sorted(v for v in neighbors[u] if v > u)
        for v, w in combinations(above, 2):
            if time.perf_counter() >= deadline or len(squares) >= limit:
                return np.asarray(squares, dtype=int).reshape((-1, 4))
            if w in neighbors[v]:
                continue
            for t in neighbors[v] & neighbors[w]:
                if t > u and t not in neighbors[u]:
                    squares.append(sorted((edge_index[u, v], edge_index[u, w],
                                           edge_index[min(v, t), max(v, t)],
                                           edge_index[min(w, t), max(w, t)])))
    return np.asarray(squares, dtype=int).reshape((-1, 4))


def separate_short_cycles(supports, x, limit=None):
    if not len(supports):
        return []
    length = supports.shape[1]
    positive = np.asarray([[bool(mask & (1 << i)) for i in range(length)]
                           for mask in range(1 << length) if mask.bit_count() % 2], dtype=int)
    signs = 2*positive-1
    rhs = positive.sum(axis=1)-1
    values = np.asarray(x)[supports]
    violation = values @ signs.T - rhs
    facet = violation.argmax(axis=1)
    best = violation[np.arange(len(facet)), facet]
    chosen = np.flatnonzero(best > 1e-7)
    chosen = chosen[np.argsort(-best[chosen], kind="stable")]
    if limit is not None:
        chosen = chosen[:limit]
    return [IsingCut(tuple(map(int, supports[i])), tuple(map(int, signs[facet[i]])),
                     int(rhs[facet[i]]), "cycle") for i in chosen]


def separate_cycles(graph, edges, x, roots, deadline, limit=256):
    """Separate odd-cycle inequalities with shortest paths in a double cover.

    The bounded root batch is a heuristic for choosing where to run exact
    shortest paths. A lack of cuts in this batch does not prove separation
    completeness. Every returned simple cycle is checked before use.
    """
    n = len(graph)
    u, v = np.asarray(edges, dtype=int).T
    x = np.clip(x, 0, 1)
    rows = np.r_[u, v, u + n, v + n, u, v + n, u + n, v]
    cols = np.r_[v, u, v + n, u + n, v + n, u, v, u + n]
    data = np.r_[x, x, x, x, 1-x, 1-x, 1-x, 1-x]
    auxiliary = sp.csr_matrix((data, (rows, cols)), shape=(2*n, 2*n))
    edge_index = {edge: i for i, edge in enumerate(edges)}
    found = {}
    for start in range(0, len(roots), 16):
        if time.perf_counter() >= deadline:
            break
        batch = np.asarray(roots[start:start+16], dtype=int)
        distances, predecessors = dijkstra(auxiliary, directed=True, indices=batch,
                                           return_predecessors=True, limit=1-1e-7)
        for row, root in enumerate(batch):
            if not np.isfinite(distances[row, root+n]):
                continue
            path, current = [], int(root+n)
            while current != root:
                path.append(current)
                current = int(predecessors[row, current])
                if current < 0:
                    raise RuntimeError("Invalid shortest-path predecessor")
            path.append(int(root))
            path.reverse()
            # A simple path in the double cover can project to a closed walk.
            # Decompose repeated original vertices into simple cycles.
            stack, choices, positions = [path[0] % n], [], {path[0] % n: 0}
            for before, after in zip(path, path[1:]):
                vertex = after % n
                edge = tuple(sorted((before % n, vertex)))
                choices.append((edge, before // n != after // n))
                if vertex in positions:
                    begin = positions[vertex]
                    cycle = stack[begin:]
                    positive = [e for e, sign in choices[begin:] if sign]
                    if len(cycle) >= 3 and len(positive) % 2:
                        cut = cycle_cut(cycle, positive, edge_index)
                        violation = sum(a*x[i] for i, a in zip(cut.indices, cut.coefficients)) - cut.rhs
                        if violation > 1e-6:
                            found[cut] = violation / np.sqrt(len(cut.indices))
                    for discarded in stack[begin+1:]:
                        positions.pop(discarded)
                    stack, choices = stack[:begin+1], choices[:begin]
                else:
                    positions[vertex] = len(stack)
                    stack.append(vertex)
        if len(found) >= limit:
            break
    return sorted(found, key=found.get, reverse=True)[:limit]


def subgraph_cut(indices, edges, weights, x, deadline, threads=0):
    """Separate the exact cut polytope of a bounded induced subgraph."""
    if not indices or time.perf_counter() >= deadline:
        return None
    support = [edges[i] for i in indices]
    assignments = cut_assignments(support)
    point = np.asarray(x)[list(indices)]
    objective = np.asarray([float(weights[edges[i]]) for i in indices])

    def integer_cut(direction):
        best, best_score = None, 1e-6
        for scale in (1, 4, 16, 64):
            coefficients = np.rint(direction * scale).astype(np.int64)
            if not np.any(coefficients):
                continue
            rhs = int((assignments @ coefficients).max())
            common = reduce(gcd, map(int, coefficients), abs(rhs)) or 1
            coefficients //= common
            rhs //= common
            violation = float(coefficients @ point) - rhs
            score = violation / max(1, np.linalg.norm(coefficients))
            if score > best_score:
                nz = np.flatnonzero(coefficients)
                best = IsingCut(tuple(indices[i] for i in nz), tuple(int(coefficients[i]) for i in nz), rhs, "subgraph")
                best_score = score
        return best

    # A local energy inequality sometimes separates without another LP.
    candidate = integer_cut(objective / max(1e-30, np.max(np.abs(objective))))
    if candidate is not None or time.perf_counter() >= deadline:
        return candidate
    # max a*x - b subject to a*cut <= b for every local cut, |a_i| <= 1.
    m = len(indices)
    session = highspy.Highs()
    for key, value in {"output_flag": False, "parallel": "off", "threads": threads,
                       "solver": "simplex", "presolve": "off"}.items():
        _check_status(session.setOptionValue(key, value))
    _check_status(session.addCols(m+1, np.r_[-point, 1.], np.r_[-np.ones(m), 0.],
                                  np.r_[np.ones(m), np.inf], 0, [], [], []))
    matrix = sp.hstack([sp.csr_matrix(assignments), -np.ones((len(assignments), 1))]).tocsr()
    _add_rows(session, matrix, np.full(len(assignments), -np.inf), np.zeros(len(assignments)))
    _check_status(session.setOptionValue("time_limit", max(0., deadline-time.perf_counter())))
    _check_status(session.run())
    solution = session.getSolution()
    if solution.value_valid:
        # We ignore the proposed rhs and recompute it by exact enumeration.
        return integer_cut(np.asarray(solution.col_value[:m]))
    return None
