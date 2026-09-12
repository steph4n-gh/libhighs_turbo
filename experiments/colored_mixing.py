"""Research: vectorized independent-set updates for the known Mixing method."""

import time

import networkx as nx
import numpy as np


def coloring(edges, n):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    colors = nx.coloring.greedy_color(graph, strategy="largest_first")
    groups = [[] for _ in range(max(colors.values(), default=-1) + 1)]
    for vertex, color in colors.items():
        groups[color].append(vertex)
    return [np.asarray(group, dtype=int) for group in groups]


def fit(matrix, vectors, groups, *, deadline, iterations=350, tolerance=1e-7):
    """Minimize the quadratic objective with simultaneous independent updates.

    There is no objective edge between members of a color group, so each
    group's updates are the ordinary coordinate minimizers. This is a known
    parallel coordinate-descent construction, not a new SDP relaxation.
    """
    blocks = [(indices, matrix[indices]) for indices in groups]
    previous = float(np.sum(vectors * (matrix @ vectors)))
    count = 0
    for count in range(iterations):
        for indices, row in blocks:
            values = -(row @ vectors)
            norms = np.linalg.norm(values, axis=1)
            active = norms > 1e-100
            vectors[indices[active]] = values[active] / norms[active, None]
        current = float(np.sum(vectors * (matrix @ vectors)))
        if previous - current < tolerance or time.perf_counter() >= deadline:
            break
        previous = current
    return vectors, dict(iterations=count + 1, numerical=current, colors=len(groups))


def projected_triangles(vectors, edges, *, limit=2048, seed=0, epsilon=0.5):
    """Propose in a small principal subspace; score in the original space."""
    from highs_turbo.ising_geometry import geometric_triangles

    started = time.perf_counter()
    _, rotation = np.linalg.eigh(vectors.T @ vectors)
    small = vectors @ rotation[:, -6:]
    small /= np.maximum(np.linalg.norm(small, axis=1)[:, None], 1e-100)
    candidates, stats = geometric_triangles(
        small, edges, limit=4 * limit, seed=seed, epsilon=epsilon
    )
    norms = np.asarray(
        [
            np.linalg.norm(sum(a * vectors[i] for i, a in triangle))
            for triangle in candidates
        ]
    )
    selected = [
        candidates[i] for i in np.argsort(norms, kind="stable") if norms[i] < 0.999
    ][:limit]
    return selected, dict(
        seconds=time.perf_counter() - started,
        candidates=len(candidates),
        projected_candidates=stats["candidates"],
    )
