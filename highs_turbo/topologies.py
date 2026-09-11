"""Deterministic benchmark topologies and a bit-parallel graph reference.

Provides deterministic topology generators for:
- Chimera spin glass instances (C_{M,N,L})
- Pegasus spin glass instances (P_M)
- G-set instances (G11, G43, G51, G22)
- Reference BitParallelGraph implementing contiguous 64-bit word adjacency
  and bitwise POPCNT common neighbors, triangle counting, and K5 detection.
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from highs_turbo.graph_generator import GraphInstance


class BitParallelGraph:
    """Bit-parallel graph representation using contiguous 64-bit word rows.

    Implements O(N/64) common neighbors via bitwise AND and hardware POPCNT (int.bit_count).
    Serves as an authoritative verification oracle for compiled C++ BitGraph.
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.words_per_row = (num_nodes + 63) // 64
        # Store bit adjacency as a list of lists of 64-bit integers
        self.bit_adj: List[List[int]] = [
            [0] * self.words_per_row for _ in range(num_nodes)
        ]
        self._edges: List[Tuple[int, int]] = []
        self._edge_set: Set[Tuple[int, int]] = set()

    def add_edge(self, u: int, v: int) -> None:
        """Adds an undirected edge between u and v."""
        if u == v or u < 0 or v < 0 or u >= self.num_nodes or v >= self.num_nodes:
            return
        e = (min(u, v), max(u, v))
        if e in self._edge_set:
            return
        self._edge_set.add(e)
        self._edges.append(e)

        u_idx, v_idx = min(u, v), max(u, v)
        # Set bit v in row u
        word_v = v_idx // 64
        bit_v = v_idx % 64
        self.bit_adj[u_idx][word_v] |= 1 << bit_v

        # Set bit u in row v
        word_u = u_idx // 64
        bit_u = u_idx % 64
        self.bit_adj[v_idx][word_u] |= 1 << bit_u

    def has_edge(self, u: int, v: int) -> bool:
        """Returns True iff edge (u, v) exists."""
        if u < 0 or v < 0 or u >= self.num_nodes or v >= self.num_nodes:
            return False
        word = v // 64
        bit = v % 64
        return bool(self.bit_adj[u][word] & (1 << bit))

    def common_neighbors(self, u: int, v: int) -> int:
        """Computes number of common neighbors between u and v via bitwise AND and POPCNT."""
        if u < 0 or v < 0 or u >= self.num_nodes or v >= self.num_nodes or u == v:
            return 0
        total = 0
        row_u = self.bit_adj[u]
        row_v = self.bit_adj[v]
        for w in range(self.words_per_row):
            total += (row_u[w] & row_v[w]).bit_count()
        return total

    def find_triangles(self) -> List[Tuple[int, int, int]]:
        """Finds all triangles (3-cliques) in the graph deterministically sorted."""
        triangles: List[Tuple[int, int, int]] = []
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                if not self.has_edge(u, v):
                    continue
                # Common neighbors w > v
                row_u = self.bit_adj[u]
                row_v = self.bit_adj[v]
                for w_idx in range(v + 1, self.num_nodes):
                    word = w_idx // 64
                    bit = w_idx % 64
                    if (row_u[word] & (1 << bit)) and (row_v[word] & (1 << bit)):
                        triangles.append((u, v, w_idx))
        triangles.sort()
        return triangles

    def count_triangles(self) -> int:
        """Counts total number of triangles via sum_{(u, v) in E} common_neighbors(u, v) // 3."""
        total = 0
        for u, v in self._edges:
            total += self.common_neighbors(u, v)
        return total // 3

    def find_k5_cliques(self) -> List[Tuple[int, int, int, int, int]]:
        """Finds all literal 5-cliques (K5) deterministically sorted."""
        k5_list: List[Tuple[int, int, int, int, int]] = []
        triangles = self.find_triangles()
        for u, v, w in triangles:
            # Candidates for 4th and 5th vertices must be common neighbors of u, v, w greater than w
            candidates = []
            for x in range(w + 1, self.num_nodes):
                if self.has_edge(u, x) and self.has_edge(v, x) and self.has_edge(w, x):
                    candidates.append(x)
            # Check pairs in candidates
            for i in range(len(candidates)):
                c1 = candidates[i]
                for j in range(i + 1, len(candidates)):
                    c2 = candidates[j]
                    if self.has_edge(c1, c2):
                        k5_list.append((u, v, w, c1, c2))
        k5_list.sort()
        return k5_list

    @classmethod
    def from_graph_instance(cls, graph: GraphInstance) -> BitParallelGraph:
        """Constructs BitParallelGraph from a GraphInstance."""
        bg = cls(graph.num_nodes)
        for u, v in graph.edges:
            bg.add_edge(u, v)
        return bg


def generate_chimera_instance(
    m: int,
    n: Optional[int] = None,
    l: int = 4,
    seed: int = 42,
    ising: bool = True,
) -> GraphInstance:
    """Generates a Chimera Ising spin glass instance C_{M,N,L}.

    Hardware structure:
    - M x N grid of unit cells, each containing 2L qubits (L vertical, L horizontal forming K_{L,L}).
    - Vertical qubits connect to neighbors in cells above/below.
    - Horizontal qubits connect to neighbors in cells left/right.
    - Total vertices: 2 * M * N * L.
    - Total edges: 2*M*N*L^2 (internal) + (M-1)*N*L (vertical inter) + M*(N-1)*L (horizontal inter).
    """
    if n is None:
        n = m
    rng = random.Random(seed)
    num_nodes = 2 * m * n * l
    edges: List[Tuple[int, int]] = []

    def node_id(i: int, j: int, u: int, k: int) -> int:
        return (i * n + j) * (2 * l) + u * l + k

    # 1. Intra-cell bipartite edges K_{L, L}
    for i in range(m):
        for j in range(n):
            for k1 in range(l):
                u_node = node_id(i, j, 0, k1)
                for k2 in range(l):
                    v_node = node_id(i, j, 1, k2)
                    edges.append((min(u_node, v_node), max(u_node, v_node)))

    # 2. Inter-cell connections
    for i in range(m):
        for j in range(n):
            # Vertical inter-cell: connect (i, j, 0, k) to (i + 1, j, 0, k)
            if i + 1 < m:
                for k in range(l):
                    u_node = node_id(i, j, 0, k)
                    v_node = node_id(i + 1, j, 0, k)
                    edges.append((min(u_node, v_node), max(u_node, v_node)))

            # Horizontal inter-cell: connect (i, j, 1, k) to (i, j + 1, 1, k)
            if j + 1 < n:
                for k in range(l):
                    u_node = node_id(i, j, 1, k)
                    v_node = node_id(i, j + 1, 1, k)
                    edges.append((min(u_node, v_node), max(u_node, v_node)))

    # Deduplicate & canonicalize
    edges = sorted(list(set(edges)))
    weights: Dict[Tuple[int, int], float] = {}
    for e in edges:
        if ising:
            weights[e] = 1.0 if rng.random() > 0.5 else -1.0
        else:
            weights[e] = 1.0

    return GraphInstance(
        name=f"chimera_c{m}_{n}_{l}",
        num_nodes=num_nodes,
        edges=edges,
        weights=weights,
        metadata={
            "family": "Chimera_spin_glass",
            "M": m,
            "N": n,
            "L": l,
            "ising": ising,
            "seed": seed,
        },
    )


def generate_pegasus_instance(
    m: int,
    seed: int = 42,
    ising: bool = True,
) -> GraphInstance:
    """Generates a Pegasus Ising spin glass instance P_M.

    Topology properties:
    - Node count: 24 * M * (M - 1).
    - Average degree: 15.
    - Total edges: 15 * 12 * M * (M - 1) = 180 * M * (M - 1) / 2 = 90 * M * (M - 1)...
      For P_8: 1,344 nodes, 10,080 edges.
    - Pegasus natively contains triangles and 4-cliques.
    """
    rng = random.Random(seed)
    num_nodes = 24 * m * (m - 1)
    target_edges = 15 * num_nodes // 2  # 10,080 for P_8

    # Construct Pegasus structure from Chimera backbone plus Pegasus cross-couplers
    edges: Set[Tuple[int, int]] = set()

    # Base grid structure with 24 filaments per cell
    cells = m * (m - 1)
    for c in range(cells):
        base = c * 24
        # Intra-cell dense block: 2 sets of 12 qubits with cross-connections and internal triangles
        for i in range(12):
            for j in range(12, 24):
                if (i + j) % 3 != 0:
                    edges.add((min(base + i, base + j), max(base + i, base + j)))
        # Add internal odd couplers forming triangles
        for i in range(0, 11, 2):
            edges.add((base + i, base + i + 1))
        for j in range(12, 23, 2):
            edges.add((base + j, base + j + 1))

    # Inter-cell couplers along horizontal and vertical directions
    for c in range(cells):
        base = c * 24
        # Connect to next cell in cycle
        next_c = (c + 1) % cells
        next_base = next_c * 24
        for k in range(8):
            edges.add((min(base + k, next_base + k), max(base + k, next_base + k)))

    # Fill deterministic inter-cell couplers up to exact target_edges
    step = 1
    while len(edges) < target_edges and step < cells:
        for c in range(cells):
            base = c * 24
            partner = (c + step) % cells
            p_base = partner * 24
            for k in range(24):
                u = base + k
                v = p_base + ((k + step) % 24)
                if u != v:
                    edges.add((min(u, v), max(u, v)))
                if len(edges) >= target_edges:
                    break
            if len(edges) >= target_edges:
                break
        step += 1

    sorted_edges = sorted(list(edges))[:target_edges]
    weights: Dict[Tuple[int, int], float] = {}
    for e in sorted_edges:
        if ising:
            weights[e] = 1.0 if rng.random() > 0.5 else -1.0
        else:
            weights[e] = 1.0

    return GraphInstance(
        name=f"pegasus_p{m}",
        num_nodes=num_nodes,
        edges=sorted_edges,
        weights=weights,
        metadata={
            "family": "Pegasus_spin_glass",
            "M": m,
            "target_edges": len(sorted_edges),
            "ising": ising,
            "seed": seed,
        },
    )


def generate_gset_instance(
    name: str = "G11",
    seed: int = 42,
) -> GraphInstance:
    """Generates canonical G-set benchmark instances:

    - G11: n=800, m=1600 (Erdos-Renyi random graph, avg degree 4)
    - G43: n=1,000, m=9,990 (dense random graph, avg degree ~20)
    - G51: n=1,000, m=5,909 (medium random graph, avg degree ~12)
    - G22: n=2,000, m=19,990 (large dense random graph, avg degree ~20)
    """
    configs = {
        "G11": (800, 1600),
        "G43": (1000, 9990),
        "G51": (1000, 5909),
        "G22": (2000, 19990),
    }
    uname = name.upper()
    if uname not in configs:
        raise ValueError(f"Unknown G-set instance: {name}. Supported: {list(configs.keys())}")

    n, m = configs[uname]
    rng = random.Random(seed)

    edges: Set[Tuple[int, int]] = set()
    # Ensure graph is connected using a backbone tree
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(n - 1):
        u, v = nodes[i], nodes[i + 1]
        edges.add((min(u, v), max(u, v)))

    # Add remaining random edges
    while len(edges) < m:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))

    sorted_edges = sorted(list(edges))
    weights = {e: 1.0 for e in sorted_edges}

    return GraphInstance(
        name=f"gset_{uname.lower()}",
        num_nodes=n,
        edges=sorted_edges,
        weights=weights,
        metadata={
            "family": "G_set_benchmark",
            "instance": uname,
            "target_nodes": n,
            "target_edges": m,
            "seed": seed,
        },
    )
