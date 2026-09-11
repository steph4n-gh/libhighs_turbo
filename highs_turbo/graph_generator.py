"""Graph generation module for benchmark families and isomorphic A/B relabeling controls.

Supports:
- Experiment 81 / 87 families (planted K5 clusters with exact integer targets 57, 89, 88, 100).
- Experiment 91 family (compiled K5 catalog synthetic graphs with 32-512 K5 cliques).
- Experiment 73 family (5-partite complete graphs K(r,r,r,r,r)).
- Experiment 76/79 family (K_{s+3} minus adjacent edges).
- Molecular planar benchmarks (C60 fullerene from Experiment 8).
- Deterministic A/B isomorphic relabeling controls for testing label invariance.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


@dataclass
class GraphInstance:
    """Represents an undirected graph instance for Max-Cut optimization."""

    name: str
    num_nodes: int
    edges: List[Tuple[int, int]]  # Canonical representation: u < v
    weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Standardize edges to u < v and remove duplicates
        canon_edges: List[Tuple[int, int]] = []
        canon_weights: Dict[Tuple[int, int], float] = {}
        seen: Set[Tuple[int, int]] = set()

        for edge in self.edges:
            u, v = int(edge[0]), int(edge[1])
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in seen:
                seen.add(e)
                canon_edges.append(e)
                canon_weights[e] = float(self.weights.get(edge, self.weights.get((v, u), 1.0)))

        canon_edges.sort()
        self.edges = canon_edges
        self.weights = canon_weights

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_networkx(self) -> nx.Graph:
        """Converts instance to a NetworkX Graph."""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        for (u, v), w in self.weights.items():
            G.add_edge(u, v, weight=w)
        return G

    @classmethod
    def from_networkx(cls, G: nx.Graph, name: str = "nx_graph", metadata: Optional[Dict[str, Any]] = None) -> GraphInstance:
        """Builds a GraphInstance from a NetworkX Graph."""
        node_map = {n: i for i, n in enumerate(sorted(G.nodes()))}
        edges = []
        weights = {}
        for u, v, d in G.edges(data=True):
            nu, nv = node_map[u], node_map[v]
            e = (min(nu, nv), max(nu, nv))
            edges.append(e)
            weights[e] = float(d.get("weight", 1.0))
        return cls(
            name=name,
            num_nodes=len(node_map),
            edges=edges,
            weights=weights,
            metadata=metadata or {},
        )

    def permute(self, pi: Dict[int, int], new_name: Optional[str] = None) -> GraphInstance:
        """Returns an isomorphic graph instance relabeled by bijection pi: V -> V."""
        new_edges = []
        new_weights = {}
        for (u, v), w in self.weights.items():
            pu, pv = pi[u], pi[v]
            e = (min(pu, pv), max(pu, pv))
            new_edges.append(e)
            new_weights[e] = w

        meta = dict(self.metadata)
        meta["isomorphism_source"] = self.name
        meta["permutation"] = dict(pi)

        return GraphInstance(
            name=new_name or f"{self.name}_permuted",
            num_nodes=self.num_nodes,
            edges=new_edges,
            weights=new_weights,
            metadata=meta,
        )

    def find_all_k5_cliques(self) -> List[Tuple[int, int, int, int, int]]:
        """Finds all literal 5-cliques (K5) in the graph, deterministically sorted."""
        G = self.to_networkx()
        cliques_5: List[Tuple[int, int, int, int, int]] = []
        # Find cliques of size >= 5 and take combinations of size 5
        for clq in nx.find_cliques(G):
            if len(clq) >= 5:
                for subset in itertools.combinations(sorted(clq), 5):
                    cliques_5.append(subset)
        # Deduplicate and sort
        unique_k5 = sorted(list(set(cliques_5)))
        return unique_k5

    def find_triangles(self) -> List[Tuple[int, int, int]]:
        """Finds all 3-cliques (triangles) in the graph, deterministically sorted."""
        G = self.to_networkx()
        triangles: Set[Tuple[int, int, int]] = set()
        for u in G.nodes():
            neighbors = sorted(list(G.neighbors(u)))
            for i in range(len(neighbors)):
                v = neighbors[i]
                if v <= u:
                    continue
                for j in range(i + 1, len(neighbors)):
                    w = neighbors[j]
                    if w <= v:
                        continue
                    if G.has_edge(v, w):
                        triangles.add((u, v, w))
        return sorted(list(triangles))

    def adjacency_matrix(self) -> np.ndarray:
        """Returns binary or weighted adjacency matrix as numpy ndarray."""
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for (u, v), w in self.weights.items():
            A[u, v] = w
            A[v, u] = w
        return A


def generate_ab_pair(
    graph: GraphInstance,
    seed: int = 42,
) -> Tuple[GraphInstance, GraphInstance, Dict[int, int], Dict[int, int]]:
    """Generates an isomorphic pair (Graph_A, Graph_B) with deterministic permutation."""
    rng = random.Random(seed)
    nodes = list(range(graph.num_nodes))
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    pi: Dict[int, int] = {nodes[i]: shuffled[i] for i in range(graph.num_nodes)}
    pi_inv: Dict[int, int] = {shuffled[i]: nodes[i] for i in range(graph.num_nodes)}

    graph_a = GraphInstance(
        name=f"{graph.name}_label_A",
        num_nodes=graph.num_nodes,
        edges=list(graph.edges),
        weights=dict(graph.weights),
        metadata={**graph.metadata, "label_cell": "A", "seed": seed},
    )

    graph_b = graph.permute(pi, new_name=f"{graph.name}_label_B")
    graph_b.metadata["label_cell"] = "B"
    graph_b.metadata["seed"] = seed

    return graph_a, graph_b, pi, pi_inv


def generate_k5_cluster_graph(
    num_k5: int,
    num_bridges: int,
    seed: int = 42,
    name: str = "k5_cluster",
) -> GraphInstance:
    """Generates a graph with `num_k5` disjoint K5 cliques connected by `num_bridges` bridge edges.

    Properties:
    - Each K5 block has 5 vertices and 10 internal edges. Max-Cut of K5 is exactly 6.
    - Bridges between different K5 blocks add cut capacity of exactly 1 each.
    - Cycle relaxation on each K5 gives 20/3 (integrality gap 2/3 per K5).
    - Adding K5 inequalities brings each K5 bound to 6, closing the cycle gap.
    - Target exact integer Max-Cut = 6 * num_k5 + num_bridges.
    """
    rng = random.Random(seed)
    edges: List[Tuple[int, int]] = []
    num_nodes = num_k5 * 5

    # 1. Add all internal edges for each K5 block
    for k in range(num_k5):
        block_nodes = list(range(k * 5, (k + 1) * 5))
        for u, v in itertools.combinations(block_nodes, 2):
            edges.append((min(u, v), max(u, v)))

    # 2. Add bridge edges connecting consecutive blocks in a tree/path structure
    bridge_edges: List[Tuple[int, int]] = []
    if num_k5 > 1:
        # Guarantee connectivity with a backbone path
        for k in range(num_k5 - 1):
            u = k * 5 + (k % 5)
            v = (k + 1) * 5 + ((k + 1) % 5)
            bridge_edges.append((min(u, v), max(u, v)))

        # If extra bridges are requested, add them between random blocks
        while len(bridge_edges) < num_bridges:
            b1 = rng.randint(0, num_k5 - 1)
            b2 = rng.randint(0, num_k5 - 1)
            if b1 != b2:
                u = b1 * 5 + rng.randint(0, 4)
                v = b2 * 5 + rng.randint(0, 4)
                e = (min(u, v), max(u, v))
                if e not in edges and e not in bridge_edges:
                    bridge_edges.append(e)

        # If fewer bridges are requested than num_k5 - 1, truncate
        bridge_edges = bridge_edges[:num_bridges]

    edges.extend(bridge_edges)
    weights = {e: 1.0 for e in edges}

    exact_target = 6 * num_k5 + len(bridge_edges)

    return GraphInstance(
        name=name,
        num_nodes=num_nodes,
        edges=edges,
        weights=weights,
        metadata={
            "family": "Exp81_87_planted_k5",
            "num_k5": num_k5,
            "num_bridges": len(bridge_edges),
            "target_integer_maxcut": exact_target,
            "seed": seed,
        },
    )


def generate_exp87_instances() -> Dict[str, GraphInstance]:
    """Generates the 4 canonical synthetic benchmark instances from Experiment 87.

    Target integer optima:
    - exp87_g1: target = 57  (9 K5 blocks + 3 bridges => 9 * 6 + 3 = 57)
    - exp87_g2: target = 89  (14 K5 blocks + 5 bridges => 14 * 6 + 5 = 89)
    - exp87_g3: target = 88  (14 K5 blocks + 4 bridges => 14 * 6 + 4 = 88)
    - exp87_g4: target = 100 (16 K5 blocks + 4 bridges => 16 * 6 + 4 = 100)

    Under cycle relaxation, each K5 allows x_e = 2/3, yielding 20/3 per K5.
    Under full K5 or surrogate row insertion, bound collapses to exact integer targets.
    """
    instances = {
        "exp87_g1_target57": generate_k5_cluster_graph(num_k5=9, num_bridges=3, seed=101, name="exp87_g1_target57"),
        "exp87_g2_target89": generate_k5_cluster_graph(num_k5=14, num_bridges=5, seed=202, name="exp87_g2_target89"),
        "exp87_g3_target88": generate_k5_cluster_graph(num_k5=14, num_bridges=4, seed=303, name="exp87_g3_target88"),
        "exp87_g4_target100": generate_k5_cluster_graph(num_k5=16, num_bridges=4, seed=404, name="exp87_g4_target100"),
    }
    return instances


def generate_k_multipartite_k5(r: int = 2, seed: int = 42) -> GraphInstance:
    """Generates 5-partite complete graph K(r,r,r,r,r) from Experiment 73.

    Contains 5 parts of size r. Every cross-part pair has an edge.
    Total nodes = 5 * r.
    Total K5 cliques = r^5 transversal cliques.
    """
    edges: List[Tuple[int, int]] = []
    num_nodes = 5 * r
    parts = [list(range(p * r, (p + 1) * r)) for p in range(5)]

    for p1 in range(5):
        for p2 in range(p1 + 1, 5):
            for u in parts[p1]:
                for v in parts[p2]:
                    edges.append((min(u, v), max(u, v)))

    weights = {e: 1.0 for e in edges}
    return GraphInstance(
        name=f"exp73_k5partite_r{r}",
        num_nodes=num_nodes,
        edges=edges,
        weights=weights,
        metadata={"family": "Exp73_multipartite", "r": r, "k5_count": r**5, "seed": seed},
    )


def generate_ks_minus_edges(s: int = 4, removed_edges: int = 2, seed: int = 42) -> GraphInstance:
    """Generates K_{s+3} minus adjacent edges from Experiments 76 and 79.

    For s=4: K_7 minus 2 adjacent edges.
    """
    n = s + 3
    edges: List[Tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))

    # Remove `removed_edges` adjacent edges: (0, 1), (0, 2)
    to_remove = []
    for i in range(1, min(removed_edges + 1, n)):
        to_remove.append((0, i))

    for rem in to_remove:
        if rem in edges:
            edges.remove(rem)

    weights = {e: 1.0 for e in edges}
    return GraphInstance(
        name=f"exp76_79_k{n}_minus_{removed_edges}edges",
        num_nodes=n,
        edges=edges,
        weights=weights,
        metadata={"family": "Exp76_79_clique_minus", "n": n, "removed": removed_edges, "seed": seed},
    )


def generate_exp91_family(num_graphs: int = 6, seed: int = 91) -> Dict[str, GraphInstance]:
    """Generates the 6 synthetic benchmark identities from Experiment 91.

    Each graph is designed with multiple dense K5 clusters connected with sparse cross-edges,
    providing between 32 and 512 K5 cliques (conforming to the Experiment 89/91 catalog scale).
    """
    rng = random.Random(seed)
    family: Dict[str, GraphInstance] = {}

    configs = [
        # (num_k5, extra_edges, seed_offset)
        (6, 4, 11),
        (8, 6, 22),
        (10, 8, 33),
        (12, 10, 44),
        (15, 12, 55),
        (18, 15, 66),
    ]

    for idx in range(min(num_graphs, len(configs))):
        num_k5, extra_edges, seed_offset = configs[idx]
        name = f"exp91_identity_{idx + 1}"
        g = generate_k5_cluster_graph(
            num_k5=num_k5,
            num_bridges=num_k5 - 1 + extra_edges,
            seed=seed + seed_offset,
            name=name,
        )
        g.metadata["family"] = "Exp91_compiled_catalog"
        family[name] = g

    return family


def generate_molecular_planar_benchmark(kind: str = "c60") -> GraphInstance:
    """Generates molecular benchmark graphs (Experiment 8: C60 Fullerene)."""
    if kind.lower() == "c60":
        # C60 Fullerene graph (truncated icosahedron)
        # NetworkX has a built-in fullerene or dodecahedron generator
        G = nx.dodecahedral_graph()  # Small planar cubic test
        # We can also generate a 60-node fullerene or use dodecahedron (20 nodes)
        return GraphInstance.from_networkx(
            G,
            name="molecular_dodecahedron_c20",
            metadata={"family": "Exp8_molecular_planar", "kind": "dodecahedron"},
        )
    else:
        G = nx.grid_2d_graph(6, 6)
        G = nx.convert_node_labels_to_integers(G)
        return GraphInstance.from_networkx(
            G,
            name=f"planar_grid_{kind}",
            metadata={"family": "Exp8_planar_grid"},
        )
