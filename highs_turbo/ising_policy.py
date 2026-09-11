"""Small learned ranking policy over current subgraph/relaxation features.

The policy only orders candidates. It cannot change an inequality, skip its
verification, or alter the certificate arithmetic. Inference uses NumPy so
the trained policy does not require a deep-learning runtime in production.
"""

import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "vertices", "density", "cycle_rank", "mean_fractionality", "max_fractionality",
    "fractional_share", "weight_mean", "weight_spread", "relaxation_satisfaction",
    "boundary_share", "degree_mean", "fractionality_spread",
)


def cluster_candidates(graph, edges, weights, x, roots, deadline):
    import time

    edge_index = {e: i for i, e in enumerate(edges)}
    absolute = np.asarray([abs(float(weights[e])) for e in edges])
    scale = max(float(absolute.max()), 1e-30)
    fractional = np.minimum(np.clip(x, 0, 1), 1-np.clip(x, 0, 1))
    candidates, features, seen = [], [], set()
    for root in roots:
        if time.perf_counter() >= deadline:
            break
        selected, frontier = [int(root)], set(graph[int(root)])
        while frontier and len(selected) < 12:
            def score(v):
                incident = [edge_index[tuple(sorted((u, v)))] for u in selected if graph.has_edge(u, v)]
                return len(incident) + sum(fractional[incident]) + 0.05*sum(absolute[incident])/scale

            chosen = max(sorted(frontier), key=score)
            selected.append(chosen)
            frontier.update(graph[chosen])
            frontier.difference_update(selected)
            if len(selected) not in (8, 12) and frontier:
                continue
            vertices = set(selected)
            indices = tuple(sorted(edge_index[tuple(sorted((u, v)))] for u in vertices
                                   for v in graph[u] if v in vertices and u < v))
            if not indices or indices in seen or len(indices) < len(vertices):
                continue
            seen.add(indices)
            f, w = fractional[list(indices)], absolute[list(indices)]
            signed = np.asarray([float(weights[edges[i]]) for i in indices])
            degrees = sum(graph.degree(v) for v in vertices)
            internal = 2*len(indices)
            features.append([
                len(vertices)/12, internal/(len(vertices)*(len(vertices)-1)),
                (len(indices)-len(vertices)+1)/len(indices), float(f.mean()), float(f.max()),
                float(np.mean(f > 1e-5)), float(w.mean())/scale, float(w.std())/scale,
                float(signed @ (2*x[list(indices)]-1))/max(float(w.sum()), 1e-30),
                (degrees-internal)/max(1, degrees), degrees/(len(vertices)*max(1, len(graph)-1)),
                float(f.std()),
            ])
            candidates.append(indices)
    return candidates, np.asarray(features, dtype=float).reshape((-1, len(FEATURE_NAMES)))


def rank_clusters(features, policy):
    if not len(features):
        return np.empty(0, dtype=int)
    if policy == "deterministic":
        # Dense, fractional supports are useful candidates for inequalities
        # beyond the cycle relaxation. This is also the learning ablation.
        scores = features[:, 1] * features[:, 3] * (1 + features[:, 2])
    elif policy == "learned":
        model = json.loads(Path(__file__).with_name("ising_policy.json").read_text())
        if model["features"] != list(FEATURE_NAMES):
            raise ValueError("Ising policy feature schema mismatch")
        standardized = (features-np.asarray(model["mean"]))/np.asarray(model["scale"])
        hidden = np.maximum(0, standardized @ np.asarray(model["w1"]) + np.asarray(model["b1"]))
        scores = hidden @ np.asarray(model["w2"]) + model["b2"]
        if not np.isfinite(scores).all():
            raise ValueError("Ising policy produced a non-finite score")
    else:
        raise ValueError("cut_policy must be 'deterministic', 'learned', or 'static'")
    return np.argsort(-scores, kind="stable")
