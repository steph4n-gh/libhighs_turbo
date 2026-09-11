"""Fast O(nnz) automatic topology scanner for highs_turbo.

Inspects (c, A_ub, b_ub, ...) to automatically identify binary quadratic and graph
cut structure, extracting the underlying graph topology to drive the compiled
C++ Neural-Surrogate Cutting Plane Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

from highs_turbo.graph_generator import GraphInstance


@dataclass
class TopologyScanResult:
    """Outcome of automatic LP topology scanning."""

    is_graph_structured: bool
    structure_type: str  # "node_edge_incidence", "triangle_metric", "qubo_mccormick", "explicit", "none"
    graph: Optional[GraphInstance] = None
    edge_var_indices: Optional[List[int]] = None
    node_var_indices: Optional[List[int]] = None
    edge_to_var: Optional[Dict[Tuple[int, int], List[int]]] = None
    confidence: float = 0.0
    num_variables: int = 0
    num_constraints: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopologyDetector:
    """Fast O(nnz) scanner for detecting combinatorial graph cut structure in LP formulations."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def detect(
        self,
        c: Sequence[float],
        A_ub: Optional[Any] = None,
        b_ub: Optional[Sequence[float]] = None,
        A_eq: Optional[Any] = None,
        b_eq: Optional[Sequence[float]] = None,
        bounds: Optional[Any] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TopologyScanResult:
        """Inspects LP formulation and extracts underlying graph if detected."""
        c_arr = np.asarray(c, dtype=np.float64).flatten()
        n_vars = len(c_arr)

        # 1. Check for explicit graph or adjacency passed via kwargs or options
        explicit_graph = kwargs.get("graph") or (options or {}).get("graph")
        if explicit_graph is not None:
            g = self._coerce_to_graph_instance(explicit_graph, c_arr)
            return TopologyScanResult(
                is_graph_structured=True,
                structure_type="explicit",
                graph=g,
                edge_var_indices=list(range(min(n_vars, g.num_edges))),
                edge_to_var={e: [i] for i, e in enumerate(g.edges)},
                confidence=1.0,
                num_variables=n_vars,
                num_constraints=0 if A_ub is None else self._get_num_rows(A_ub),
                metadata={"source": "explicit_graph"},
            )

        explicit_adj = kwargs.get("adj") or kwargs.get("adjacency") or (options or {}).get("adj")
        if explicit_adj is not None:
            g = self._graph_from_adjacency(explicit_adj)
            return TopologyScanResult(
                is_graph_structured=True,
                structure_type="explicit",
                graph=g,
                edge_var_indices=list(range(min(n_vars, g.num_edges))),
                edge_to_var={e: [i] for i, e in enumerate(g.edges)},
                confidence=1.0,
                num_variables=n_vars,
                num_constraints=0 if A_ub is None else self._get_num_rows(A_ub),
                metadata={"source": "explicit_adjacency"},
            )

        if A_ub is None:
            # Unconstrained over variables or bounding box only
            return TopologyScanResult(
                is_graph_structured=False,
                structure_type="none",
                num_variables=n_vars,
                num_constraints=0,
            )

        num_rows = self._get_num_rows(A_ub)
        if num_rows == 0:
            return TopologyScanResult(
                is_graph_structured=False,
                structure_type="none",
                num_variables=n_vars,
                num_constraints=0,
            )

        # Convert A_ub to CSR representation for fast O(nnz) row scanning
        csr_A, b_vec = self._to_csr_and_vector(A_ub, b_ub, num_rows, n_vars)

        # 2. Check Pattern A: Node-Edge Incidence (Standard Max-Cut MILP/LP formulation)
        node_edge_res = self._scan_node_edge_incidence(c_arr, csr_A, b_vec, n_vars, num_rows)
        if node_edge_res is not None:
            return node_edge_res

        # 3. Check Pattern B: Triangle / Metric Cycle Inequalities on Edge Variables
        triangle_res = self._scan_triangle_inequalities(c_arr, csr_A, b_vec, n_vars, num_rows)
        if triangle_res is not None:
            return triangle_res

        # 4. Check Pattern C: McCormick Envelopes (QUBO / Quadratic Binary Linearization)
        mccormick_res = self._scan_mccormick_envelopes(c_arr, csr_A, b_vec, n_vars, num_rows)
        if mccormick_res is not None:
            return mccormick_res

        # Non-graph standard LP
        return TopologyScanResult(
            is_graph_structured=False,
            structure_type="none",
            num_variables=n_vars,
            num_constraints=num_rows,
        )

    def _get_num_rows(self, A: Any) -> int:
        if hasattr(A, "shape"):
            return int(A.shape[0])
        if isinstance(A, (list, tuple)):
            return len(A)
        return 0

    def _to_csr_and_vector(
        self, A: Any, b: Optional[Any], num_rows: int, n_vars: int
    ) -> Tuple[Any, np.ndarray]:
        """Converts A to CSR format and b to 1D array."""
        b_vec = np.zeros(num_rows, dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64).flatten()
        if sp is not None and sp.issparse(A):
            return A.tocsr(), b_vec
        arr = np.asarray(A, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if sp is not None:
            return sp.csr_matrix(arr), b_vec
        # Fallback minimal mock CSR filtering nonzeros
        class DenseAsCSR:
            def __init__(self, mat: np.ndarray, tol: float = 1e-9):
                self.mat = mat
                self.shape = mat.shape
                indptr = [0]
                indices = []
                data = []
                for r in range(mat.shape[0]):
                    nz_cols = np.nonzero(np.abs(mat[r]) > tol)[0]
                    indices.extend(nz_cols.tolist())
                    data.extend(mat[r, nz_cols].tolist())
                    indptr.append(len(indices))
                self.indptr = np.array(indptr, dtype=int)
                self.indices = np.array(indices, dtype=int)
                self.data = np.array(data, dtype=np.float64)

        return DenseAsCSR(arr), b_vec

    def _scan_node_edge_incidence(
        self, c: np.ndarray, A_csr: Any, b_vec: np.ndarray, n_vars: int, num_rows: int
    ) -> Optional[TopologyScanResult]:
        """Scans for x_i - s_u - s_v <= 0 patterns connecting edge variables to node variables."""
        edge_to_nodes: Dict[int, List[Tuple[int, int]]] = {}
        node_var_set: Set[int] = set()
        edge_var_set: Set[int] = set()

        indptr = A_csr.indptr
        indices = A_csr.indices
        data = A_csr.data

        match_count = 0
        for r in range(num_rows):
            start = indptr[r]
            end = indptr[r + 1]
            nnz = end - start
            if nnz != 3:
                continue

            rhs = b_vec[r] if r < len(b_vec) else 0.0
            row_idx = indices[start:end]
            row_vals = data[start:end]

            # Look for: 1 positive (+1) and 2 negative (-1), with RHS <= 0
            if abs(rhs) <= self.tolerance:
                pos = [row_idx[k] for k in range(3) if abs(row_vals[k] - 1.0) <= self.tolerance]
                neg = [row_idx[k] for k in range(3) if abs(row_vals[k] + 1.0) <= self.tolerance]
                if len(pos) == 1 and len(neg) == 2:
                    e_idx = pos[0]
                    u_idx, v_idx = min(neg[0], neg[1]), max(neg[0], neg[1])
                    if e_idx not in edge_to_nodes:
                        edge_to_nodes[e_idx] = []
                    # Check if (u_idx, v_idx) is already added
                    if (u_idx, v_idx) not in edge_to_nodes[e_idx]:
                        edge_to_nodes[e_idx].append((u_idx, v_idx))
                    edge_var_set.add(e_idx)
                    node_var_set.add(u_idx)
                    node_var_set.add(v_idx)
                    match_count += 1

        if len(edge_to_nodes) >= 3 and match_count >= 3:
            # Found node-edge incidence pattern!
            # Relabel nodes to contiguous 0..N-1
            sorted_nodes = sorted(list(node_var_set))
            node_map = {orig: i for i, orig in enumerate(sorted_nodes)}

            edges: List[Tuple[int, int]] = []
            weights: Dict[Tuple[int, int], float] = {}
            edge_to_var: Dict[Tuple[int, int], List[int]] = {}
            sorted_edge_vars = sorted(list(edge_to_nodes.keys()))

            for e_var in sorted_edge_vars:
                # We can only use variables that correspond to exactly 1 edge
                if len(edge_to_nodes[e_var]) != 1:
                    continue
                u_orig, v_orig = edge_to_nodes[e_var][0]
                u_canon, v_canon = node_map[u_orig], node_map[v_orig]
                e_canon = (min(u_canon, v_canon), max(u_canon, v_canon))
                edges.append(e_canon)
                # In Max-Cut: c_e = -w_e
                w = -c[e_var] if e_var < len(c) else 1.0
                weights[e_canon] = weights.get(e_canon, 0.0) + float(w)
                if e_canon not in edge_to_var:
                    edge_to_var[e_canon] = []
                edge_to_var[e_canon].append(e_var)

            graph = GraphInstance(
                name=f"detected_incidence_graph_{len(sorted_nodes)}n_{len(edges)}e",
                num_nodes=len(sorted_nodes),
                edges=edges,
                weights=weights,
                metadata={"detected_from": "node_edge_incidence"},
            )

            return TopologyScanResult(
                is_graph_structured=True,
                structure_type="node_edge_incidence",
                graph=graph,
                edge_var_indices=sorted_edge_vars,
                node_var_indices=sorted_nodes,
                edge_to_var=edge_to_var,
                confidence=min(1.0, len(edge_to_nodes) / max(1, n_vars // 2)),
                num_variables=n_vars,
                num_constraints=num_rows,
            )

        return None

    def _scan_triangle_inequalities(
        self, c: np.ndarray, A_csr: Any, b_vec: np.ndarray, n_vars: int, num_rows: int
    ) -> Optional[TopologyScanResult]:
        """Scans for triangle cycle inequalities on edge variables."""
        indptr = A_csr.indptr
        indices = A_csr.indices
        data = A_csr.data

        triangles: Set[Tuple[int, int, int]] = set()
        matched_rows = 0

        for r in range(num_rows):
            start = indptr[r]
            end = indptr[r + 1]
            nnz = end - start
            if nnz != 3:
                continue

            rhs = b_vec[r] if r < len(b_vec) else 0.0
            # Triangle inequality RHS is 1.0 (odd cycle) or 2.0 (all positive)
            if abs(rhs - 1.0) > self.tolerance and abs(rhs - 2.0) > self.tolerance:
                continue

            row_idx = indices[start:end]
            row_vals = data[start:end]

            # All coefficients must be +/- 1.0
            if all(abs(abs(v) - 1.0) <= self.tolerance for v in row_vals):
                tri = tuple(sorted([int(row_idx[0]), int(row_idx[1]), int(row_idx[2])]))
                triangles.add(tri)
                matched_rows += 1

        # A triangle LP has multiple rows per triangle (up to 4 per triangle)
        if len(triangles) >= 1 and matched_rows >= 3:
            graph, edge_to_var = self._reconstruct_graph_from_triangles(triangles, c, n_vars)
            if graph is not None:
                return TopologyScanResult(
                    is_graph_structured=True,
                    structure_type="triangle_metric",
                    graph=graph,
                    edge_var_indices=list(range(n_vars)),
                    edge_to_var=edge_to_var,
                    confidence=min(1.0, len(triangles) * 4 / max(1, num_rows)),
                    num_variables=n_vars,
                    num_constraints=num_rows,
                )

        return None

    def _reconstruct_graph_from_triangles(
        self, triangles: Set[Tuple[int, int, int]], c: np.ndarray, n_vars: int
    ) -> Tuple[Optional[GraphInstance], Dict[Tuple[int, int], int]]:
        """Reconstructs vertex assignment (u, v) for edge variables using triangle cycle incidence."""
        from collections import defaultdict

        m = n_vars
        edge_adj: Dict[int, Set[int]] = defaultdict(set)
        for e1, e2, e3 in triangles:
            edge_adj[e1].add(e2)
            edge_adj[e1].add(e3)
            edge_adj[e2].add(e1)
            edge_adj[e2].add(e3)
            edge_adj[e3].add(e1)
            edge_adj[e3].add(e2)

        visited: Set[int] = set()
        components: List[List[int]] = []
        for e in sorted(edge_adj.keys()):
            if e not in visited:
                comp: List[int] = []
                q = [e]
                visited.add(e)
                while q:
                    cur = q.pop(0)
                    comp.append(cur)
                    for nb in edge_adj[cur]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                components.append(sorted(comp))

        next_vertex = 0
        edge_endpoints: Dict[int, Tuple[int, int]] = {}

        for comp in components:
            comp_set = set(comp)
            comp_tri = [t for t in triangles if t[0] in comp_set and t[1] in comp_set and t[2] in comp_set]
            if len(comp) == 10 and len(comp_tri) == 10:
                # Planted K5 clique motif
                e0 = comp[0]
                e0_triangles = [t for t in comp_tri if e0 in t]
                if len(e0_triangles) == 3:
                    t1, t2, t3 = e0_triangles
                    t_edges = set()
                    for t in e0_triangles:
                        t_edges.update(t)
                    rem_edges = list(set(comp) - t_edges)

                    p_t1 = [e for e in t1 if e != e0]
                    a1, b1 = p_t1[0], p_t1[1]
                    p_t2 = [e for e in t2 if e != e0]
                    p_t3 = [e for e in t3 if e != e0]

                    a2, b2, r12 = None, None, None
                    for cand in p_t2:
                        for r in rem_edges:
                            if tuple(sorted([a1, cand, r])) in comp_tri:
                                a2 = cand
                                b2 = p_t2[1] if p_t2[0] == cand else p_t2[0]
                                r12 = r
                                break
                        if a2 is not None:
                            break

                    a3, b3, r13 = None, None, None
                    for cand in p_t3:
                        for r in rem_edges:
                            if tuple(sorted([a1, cand, r])) in comp_tri:
                                a3 = cand
                                b3 = p_t3[1] if p_t3[0] == cand else p_t3[0]
                                r13 = r
                                break
                        if a3 is not None:
                            break

                    if a2 is not None and a3 is not None and r12 is not None and r13 is not None:
                        r23_candidates = [r for r in rem_edges if r != r12 and r != r13]
                        if r23_candidates:
                            r23 = r23_candidates[0]
                            v0 = next_vertex
                            v1 = next_vertex + 1
                            v2 = next_vertex + 2
                            v3 = next_vertex + 3
                            v4 = next_vertex + 4
                            next_vertex += 5

                            edge_endpoints[e0] = (v0, v1)
                            edge_endpoints[a1] = (v0, v2)
                            edge_endpoints[a2] = (v0, v3)
                            edge_endpoints[a3] = (v0, v4)
                            edge_endpoints[b1] = (v1, v2)
                            edge_endpoints[b2] = (v1, v3)
                            edge_endpoints[b3] = (v1, v4)
                            edge_endpoints[r12] = (v2, v3)
                            edge_endpoints[r13] = (v2, v4)
                            edge_endpoints[r23] = (v3, v4)
                            continue

            # General triangle reconstruction for non-K5 or arbitrary motifs
            comp_edges = sorted(list(comp))
            comp_parent = {2 * e: 2 * e for e in comp_edges}
            comp_parent.update({2 * e + 1: 2 * e + 1 for e in comp_edges})

            def comp_find(x: int) -> int:
                if comp_parent[x] == x:
                    return x
                comp_parent[x] = comp_find(comp_parent[x])
                return comp_parent[x]

            def comp_union(x: int, y: int) -> None:
                rx, ry = comp_find(x), comp_find(y)
                if rx != ry:
                    comp_parent[rx] = ry

            e_to_tri: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for e1, e2, e3 in comp_tri:
                e_to_tri[e1].append((e2, e3))
                e_to_tri[e2].append((e1, e3))
                e_to_tri[e3].append((e1, e2))

            for e in comp_edges:
                pairs = e_to_tri[e]
                if not pairs:
                    continue
                p_adj: Dict[int, Set[int]] = defaultdict(set)
                for ep, edp in pairs:
                    p_adj[ep].add(edp)
                    p_adj[edp].add(ep)
                color: Dict[int, int] = {}
                for sn in p_adj:
                    if sn not in color:
                        color[sn] = 0
                        q = [sn]
                        while q:
                            cur = q.pop(0)
                            for nb in p_adj[cur]:
                                if nb not in color:
                                    color[nb] = 1 - color[cur]
                                    q.append(nb)
                for ep, clr in color.items():
                    ep_pairs = e_to_tri[ep]
                    ep_adj: Dict[int, Set[int]] = defaultdict(set)
                    for a, b_e in ep_pairs:
                        ep_adj[a].add(b_e)
                        ep_adj[b_e].add(a)
                    ep_color: Dict[int, int] = {}
                    for sn in ep_adj:
                        if sn not in ep_color:
                            ep_color[sn] = 0
                            q = [sn]
                            while q:
                                cur = q.pop(0)
                                for nb in ep_adj[cur]:
                                    if nb not in ep_color:
                                        ep_color[nb] = 1 - ep_color[cur]
                                        q.append(nb)
                    ep_side = ep_color.get(e, 0)
                    comp_union(2 * e + clr, 2 * ep + ep_side)

            comp_nodes = sorted(list(set(comp_find(2 * e) for e in comp_edges) | set(comp_find(2 * e + 1) for e in comp_edges)))
            comp_map = {orig: next_vertex + i for i, orig in enumerate(comp_nodes)}
            next_vertex += len(comp_nodes)

            for e in comp_edges:
                edge_endpoints[e] = (comp_map[comp_find(2 * e)], comp_map[comp_find(2 * e + 1)])

        # Handle any edge variables 0..n_vars-1 not present in triangles
        for e in range(n_vars):
            if e not in edge_endpoints:
                edge_endpoints[e] = (next_vertex, next_vertex + 1)
                next_vertex += 2

        mapped_edges = []
        edge_to_var = {}
        for e in range(n_vars):
            u, v = edge_endpoints[e]
            canon_e = (min(u, v), max(u, v))
            mapped_edges.append(canon_e)
            if canon_e not in edge_to_var:
                edge_to_var[canon_e] = []
            edge_to_var[canon_e].append(e)

        weights = {canon_e: float(-c[e]) if e < len(c) else 1.0 for e, canon_e in enumerate(mapped_edges)}

        graph = GraphInstance(
            name=f"detected_triangle_graph_{next_vertex}n_{len(mapped_edges)}e",
            num_nodes=next_vertex,
            edges=mapped_edges,
            weights=weights,
            metadata={"detected_from": "triangle_metric"},
        )
        return graph, edge_to_var

    def _scan_mccormick_envelopes(
        self, c: np.ndarray, A_csr: Any, b_vec: np.ndarray, n_vars: int, num_rows: int
    ) -> Optional[TopologyScanResult]:
        """Scans for McCormick envelope bilinear relaxation rows y_ij - x_i <= 0."""
        indptr = A_csr.indptr
        indices = A_csr.indices
        data = A_csr.data

        quad_to_lin: Dict[int, List[int]] = {}

        for r in range(num_rows):
            start = indptr[r]
            end = indptr[r + 1]
            nnz = end - start
            if nnz != 2:
                continue

            rhs = b_vec[r] if r < len(b_vec) else 0.0
            if abs(rhs) > self.tolerance:
                continue

            row_idx = indices[start:end]
            row_vals = data[start:end]

            if abs(row_vals[0] - 1.0) <= self.tolerance and abs(row_vals[1] + 1.0) <= self.tolerance:
                y_var, x_var = int(row_idx[0]), int(row_idx[1])
            elif abs(row_vals[1] - 1.0) <= self.tolerance and abs(row_vals[0] + 1.0) <= self.tolerance:
                y_var, x_var = int(row_idx[1]), int(row_idx[0])
            else:
                continue

            if y_var not in quad_to_lin:
                quad_to_lin[y_var] = []
            if x_var not in quad_to_lin[y_var]:
                quad_to_lin[y_var].append(x_var)

        valid_quads = {y: xs for y, xs in quad_to_lin.items() if len(xs) == 2}
        if len(valid_quads) >= 2:
            all_nodes = set()
            for xs in valid_quads.values():
                all_nodes.add(xs[0])
                all_nodes.add(xs[1])

            sorted_nodes = sorted(list(all_nodes))
            node_map = {orig: i for i, orig in enumerate(sorted_nodes)}

            edges = []
            weights = {}
            edge_to_var = {}
            for y_var, (u_orig, v_orig) in valid_quads.items():
                u = node_map[u_orig]
                v = node_map[v_orig]
                e = (min(u, v), max(u, v))
                edges.append(e)
                w = float(c[y_var]) if y_var < len(c) else 1.0
                weights[e] = weights.get(e, 0.0) + w
                if e not in edge_to_var:
                    edge_to_var[e] = []
                edge_to_var[e].append(y_var)

            graph = GraphInstance(
                name=f"detected_qubo_graph_{len(sorted_nodes)}n_{len(edges)}e",
                num_nodes=len(sorted_nodes),
                edges=edges,
                weights=weights,
                metadata={"detected_from": "qubo_mccormick"},
            )

            return TopologyScanResult(
                is_graph_structured=True,
                structure_type="qubo_mccormick",
                graph=graph,
                edge_var_indices=list(valid_quads.keys()),
                node_var_indices=sorted_nodes,
                edge_to_var=edge_to_var,
                confidence=min(1.0, len(valid_quads) / max(1, n_vars // 2)),
                num_variables=n_vars,
                num_constraints=num_rows,
            )

        return None

    def _coerce_to_graph_instance(self, obj: Any, c: np.ndarray) -> GraphInstance:
        """Coerces various graph representations to a GraphInstance."""
        if isinstance(obj, GraphInstance):
            return obj
        if hasattr(obj, "nodes") and hasattr(obj, "edges"):
            nx_nodes = list(obj.nodes())
            node_map = {n: i for i, n in enumerate(nx_nodes)}
            edges = []
            weights = {}
            for u, v, data in obj.edges(data=True):
                cu, cv = node_map[u], node_map[v]
                e = (min(cu, cv), max(cu, cv))
                edges.append(e)
                weights[e] = float(data.get("weight", 1.0))
            return GraphInstance(
                name=getattr(obj, "name", "networkx_graph"),
                num_nodes=len(nx_nodes),
                edges=edges,
                weights=weights,
            )
        return self._graph_from_adjacency(obj)

    def _graph_from_adjacency(self, adj: Any) -> GraphInstance:
        """Constructs GraphInstance from 2D adjacency matrix or sparse matrix."""
        if sp is not None and sp.issparse(adj):
            adj_csr = adj.tocsr()
            n = adj_csr.shape[0]
            edges = []
            weights = {}
            for i in range(n):
                for j_idx in range(adj_csr.indptr[i], adj_csr.indptr[i + 1]):
                    j = int(adj_csr.indices[j_idx])
                    if i < j:
                        w = float(adj_csr.data[j_idx])
                        edges.append((i, j))
                        weights[(i, j)] = w
            return GraphInstance(
                name=f"sparse_adj_graph_{n}n_{len(edges)}e",
                num_nodes=n,
                edges=edges,
                weights=weights,
            )

        arr = np.asarray(adj, dtype=np.float64)
        n = arr.shape[0]
        edges = []
        weights = {}
        for i in range(n):
            for j in range(i + 1, n):
                val = float(arr[i, j])
                if abs(val) > self.tolerance:
                    edges.append((i, j))
                    weights[(i, j)] = val

        return GraphInstance(
            name=f"dense_adj_graph_{n}n_{len(edges)}e",
            num_nodes=n,
            edges=edges,
            weights=weights,
        )


def detect_topology(
    c: Sequence[float],
    A_ub: Optional[Any] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Any] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Optional[Any] = None,
    options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> TopologyScanResult:
    """Convenience function for scanning LP topology in O(nnz) time."""
    detector = TopologyDetector()
    return detector.detect(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        options=options,
        **kwargs,
    )
