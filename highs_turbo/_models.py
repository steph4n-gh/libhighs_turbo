"""Exact normalization shared by public solvers and certificate workflows."""

from fractions import Fraction

import networkx as nx
import numpy as np
import scipy.sparse as sp

from highs_turbo.graph_generator import GraphInstance


def _maxcut_weight(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if np.iscomplexobj(value):
        raise ValueError("Max-Cut weights must be finite real numbers")
    try:
        return Fraction(float(value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Max-Cut weights must be finite real numbers") from error


def normalize_maxcut(model):
    """Return ordered labels and exact weights keyed by integer pairs u < v.

    Matrices retain the existing strict-upper-triangle convention. Stored
    duplicate upper entries and parallel undirected graph edges add exactly;
    the lower triangle and self-loops contribute nothing to the cut. Every
    supplied matrix entry or graph-edge weight must nevertheless be finite
    and real. Fraction coefficients retain their exact values.
    """
    weights = {}

    def add(u, v, raw):
        value = _maxcut_weight(raw)
        if u != v:
            edge = (min(u, v), max(u, v))
            weights[edge] = weights.get(edge, Fraction()) + value

    if isinstance(model, GraphInstance):
        n = model.num_nodes
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("GraphInstance num_nodes must be a nonnegative integer")
        labels = tuple(range(n))
        seen = set()
        for edge in model.edges:
            if (not isinstance(edge, (tuple, list)) or len(edge) != 2
                    or any(isinstance(v, (bool, np.bool_))
                           or not isinstance(v, (int, np.integer)) or not 0 <= v < n
                           for v in edge)):
                raise ValueError("GraphInstance edges must reference existing integer nodes")
            u, v = map(int, edge)
            canonical = (min(u, v), max(u, v))
            # GraphInstance is a simple graph, already deduplicated by its
            # constructor. Repeated edge listings do not introduce multiedges.
            if canonical not in seen:
                seen.add(canonical)
                add(u, v, model.weights.get((u, v), model.weights.get((v, u), 1.)))
    elif isinstance(model, nx.Graph):
        if model.is_directed():
            raise ValueError("Max-Cut requires an undirected graph")
        labels = tuple(model.nodes())
        index = {label: i for i, label in enumerate(labels)}
        for u, v, data in model.edges(data=True):
            add(index[u], index[v], data.get("weight", 1.))
    else:
        if sp.issparse(model):
            if model.ndim != 2 or model.shape[0] != model.shape[1]:
                raise ValueError("Max-Cut adjacency must be a square matrix")
            labels = tuple(range(model.shape[0]))
            # COO exposes stored duplicates without floating-point summation.
            stored = model.tocoo(copy=False)
            entries = zip(stored.row, stored.col, stored.data)
        else:
            try:
                array = np.asarray(model)
            except (TypeError, ValueError) as error:
                raise ValueError("Max-Cut adjacency must be a square matrix") from error
            if array.ndim != 2 or array.shape[0] != array.shape[1]:
                raise ValueError("Max-Cut adjacency must be a square matrix")
            labels = tuple(range(array.shape[0]))
            if array.dtype.kind == "O":
                entries = ((i, j, value) for (i, j), value in np.ndenumerate(array))
            else:
                if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                    raise ValueError("Max-Cut weights must be finite real numbers")
                rows, columns = np.nonzero(array)
                entries = zip(rows, columns, array[rows, columns])
        for u, v, raw in entries:
            if not (0 <= u < len(labels) and 0 <= v < len(labels)):
                raise ValueError("Max-Cut adjacency indices must reference existing nodes")
            value = _maxcut_weight(raw)
            if u < v:
                add(int(u), int(v), value)
    return labels, {edge: value for edge, value in sorted(weights.items()) if value}


def qubo_to_ising(num_variables, coefficients, offset=Fraction()):
    """Convert validated exact ``(i, j, value)`` terms using x=(1+s)/2.

    Every stored term contributes, including duplicates and opposite orientations.
    Callers validate dimensions and convert coefficients to Fraction first.
    """
    fields = dict.fromkeys(range(num_variables), Fraction())
    constant = offset
    couplings = {}
    for i, j, value in coefficients:
        if i == j:
            fields[i] += value / 2
            constant += value / 2
        elif value:
            pair = (min(i, j), max(i, j))
            couplings[pair] = couplings.get(pair, Fraction()) + value / 4
    couplings = {pair: value for pair, value in sorted(couplings.items()) if value}
    for (i, j), value in couplings.items():
        fields[i] += value
        fields[j] += value
        constant += value
    return fields, couplings, constant
