"""Edge-Equivariant Graph Neural Network for predictive surrogate dual aggregation.

Features:
- Topological feature extractor including Harmonic Edge Rarity from Experiment 92c.
- Strictly permutation-equivariant message passing over undirected graphs.
- Symmetric edge representations ensuring e_{uv} == e_{vu}.
- Clique pooling head predicting non-negative surrogate cutting plane multipliers.
- Face-invariant target training support.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from highs_turbo.graph_generator import GraphInstance


def extract_topological_features(graph: GraphInstance) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], int]]:
    """Extracts node and edge topological features from a GraphInstance.

    Edge features include Harmonic Edge Rarity (Exp 92c):
        R(u, v) = (1 / (1 + T(u, v))) * (2 * d_u * d_v / (d_u + d_v))
    Returns:
    - node_features: Tensor of shape (num_nodes, num_node_feats)
    - edge_features: Tensor of shape (num_edges, num_edge_feats)
    - edge_to_idx: Dict mapping sorted edge (u, v) to index in edge_features
    """
    try:
        from highs_turbo.compiled_engine import COMPILED_ENGINE_AVAILABLE, CompiledBitGraph
        if COMPILED_ENGINE_AVAILABLE:
            cbg = CompiledBitGraph.from_graph_instance(graph)
            edges = list(graph.edges)
            weights = [float(graph.weights.get(e, 1.0)) for e in edges]
            node_np, edge_np = cbg.extract_topological_features(edges, weights)
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            return torch.from_numpy(node_np), torch.from_numpy(edge_np), edge_to_idx
    except Exception:
        pass

    n = graph.num_nodes
    m = graph.num_edges
    G = graph.to_networkx()

    # Precompute degrees and triangles
    degrees = dict(G.degree())
    triangles_per_node = nx.triangles(G)
    clustering = nx.clustering(G)

    k5_cliques = graph.find_all_k5_cliques()
    k5_count_per_node: Dict[int, int] = {i: 0 for i in range(n)}
    k5_count_per_edge: Dict[Tuple[int, int], int] = {e: 0 for e in graph.edges}

    for k5 in k5_cliques:
        for node in k5:
            k5_count_per_node[node] += 1
        for u, v in itertools.combinations(k5, 2):
            e = (min(u, v), max(u, v))
            if e in k5_count_per_edge:
                k5_count_per_edge[e] += 1

    # 1. Node Features (5 dims):
    # - Normalized degree
    # - Triangle participation
    # - K5 participation
    # - Clustering coefficient
    # - Log degree
    node_feat_list = []
    max_possible_triangles = max(1, (n - 1) * (n - 2) // 2)
    for u in range(n):
        deg = degrees.get(u, 0)
        norm_deg = deg / max(1, n - 1)
        norm_tri = triangles_per_node.get(u, 0) / max_possible_triangles
        norm_k5 = k5_count_per_node.get(u, 0) / max(1, len(k5_cliques)) if k5_cliques else 0.0
        cc = clustering.get(u, 0.0)
        log_deg = math.log1p(deg)
        node_feat_list.append([norm_deg, norm_tri, norm_k5, cc, log_deg])

    if node_feat_list:
        node_features = torch.tensor(node_feat_list, dtype=torch.float32)
    else:
        node_features = torch.empty((0, 5), dtype=torch.float32)

    # 2. Edge Features (8 dims):
    # - Harmonic Degree: 2 * du * dv / (du + dv)
    # - Harmonic Edge Rarity (Exp 92c): (1 / (1 + T(u, v))) * (2 * du * dv / (du + dv))
    # - Triangle Count T(u, v) = |N(u) cap N(v)|
    # - Jaccard Coefficient: |N(u) cap N(v)| / |N(u) cup N(v)|
    # - Adamic-Adar index
    # - K5 participation
    # - Edge weight w_e
    # - Degree difference |du - dv|
    edge_feat_list = []
    edge_to_idx = {}

    for idx, (u, v) in enumerate(graph.edges):
        edge_to_idx[(u, v)] = idx
        du, dv = degrees.get(u, 0), degrees.get(v, 0)

        # Common neighbors
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        common = nu.intersection(nv)
        union = nu.union(nv)
        num_common = len(common)

        # Harmonic degree
        if du + dv > 0:
            harm_deg = (2.0 * du * dv) / (du + dv)
        else:
            harm_deg = 0.0

        # Harmonic edge rarity (Exp 92c)
        harm_rarity = (1.0 / (1.0 + num_common)) * harm_deg

        # Jaccard
        jaccard = num_common / max(1, len(union))

        # Adamic-Adar
        aa = sum(1.0 / math.log(degrees.get(w, 2) + 1.1) for w in common)

        # K5 count
        k5_part = float(k5_count_per_edge.get((u, v), 0))

        # Weight
        w = float(graph.weights.get((u, v), 1.0))

        deg_diff = abs(du - dv)

        edge_feat_list.append([
            harm_deg,
            harm_rarity,
            float(num_common),
            jaccard,
            aa,
            k5_part,
            w,
            float(deg_diff),
        ])

    if edge_feat_list:
        edge_features = torch.tensor(edge_feat_list, dtype=torch.float32)
    else:
        edge_features = torch.empty((0, 8), dtype=torch.float32)
    return node_features, edge_features, edge_to_idx


class EquivariantMessagePassingLayer(nn.Module):
    """Permutation-equivariant message passing layer with symmetric edge updates."""

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

        self.node_update = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        h_node: torch.Tensor,
        h_edge: torch.Tensor,
        edge_index: torch.Tensor,  # Shape (2, m) where edge_index[0, i] < edge_index[1, i]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_nodes = edge_index[0]
        v_nodes = edge_index[1]

        # Bidirectional message computation
        # Message u -> v
        in_uv = torch.cat([h_node[u_nodes], h_node[v_nodes], h_edge], dim=-1)
        m_uv = self.msg_mlp(in_uv)

        # Message v -> u
        in_vu = torch.cat([h_node[v_nodes], h_node[u_nodes], h_edge], dim=-1)
        m_vu = self.msg_mlp(in_vu)

        # Aggregate incoming messages per node
        n = h_node.shape[0]
        agg_msgs = torch.zeros_like(h_node)
        agg_msgs.index_add_(0, v_nodes, m_uv)
        agg_msgs.index_add_(0, u_nodes, m_vu)

        # Residual node update
        new_h_node = h_node + self.node_update(torch.cat([h_node, agg_msgs], dim=-1))

        # Symmetrized edge update using sum and abs-diff: invariant to u <-> v
        node_sum = new_h_node[u_nodes] + new_h_node[v_nodes]
        node_diff = torch.abs(new_h_node[u_nodes] - new_h_node[v_nodes])
        edge_in = torch.cat([h_edge, node_sum, node_diff], dim=-1)
        new_h_edge = h_edge + self.edge_update(edge_in)

        return new_h_node, new_h_edge


class EdgeEquivariantSurrogateGNN(nn.Module):
    """Edge-Equivariant GNN predicting candidate surrogate cutting plane multipliers."""

    def __init__(
        self,
        node_in_dim: int = 5,
        edge_in_dim: int = 8,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EquivariantMessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Head to score 5-cliques from pooled edge features
        self.clique_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensures non-negative multipliers for conic validity!
        )

        # Direct edge-level scoring head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        graph: GraphInstance,
        k5_cliques: Optional[List[Tuple[int, int, int, int, int]]] = None,
    ) -> Dict[str, Any]:
        """Runs forward pass on graph and predicts surrogate multipliers.

        Returns dict with:
        - k5_multipliers: Dict[clique_tuple, float]
        - candidate_row: np.ndarray of edge weights
        - candidate_rhs: float
        """
        if k5_cliques is None:
            k5_cliques = graph.find_all_k5_cliques()

        node_feats, edge_feats, edge_to_idx = extract_topological_features(graph)

        m = graph.num_edges
        if m == 0:
            return {
                "k5_multipliers": {},
                "k5_tensor_multipliers": {},
                "candidate_row": np.zeros(0),
                "candidate_rhs": 0.0,
            }

        edge_index = torch.tensor([[e[0] for e in graph.edges], [e[1] for e in graph.edges]], dtype=torch.long)

        h_n = self.node_encoder(node_feats)
        h_e = self.edge_encoder(edge_feats)

        for layer in self.layers:
            h_n, h_e = layer(h_n, h_e, edge_index)

        # Edge-level predictions
        edge_mults = self.edge_head(h_e).squeeze(-1)

        # Clique-level predictions: pool 10 edge representations for each K5
        k5_multipliers: Dict[Tuple[int, int, int, int, int], float] = {}
        k5_tensor_multipliers: Dict[Tuple[int, int, int, int, int], torch.Tensor] = {}
        candidate_row = np.zeros(m, dtype=np.float64)
        candidate_rhs = 0.0

        if k5_cliques:
            clique_edge_tensors = []
            valid_cliques = []
            for clq in k5_cliques:
                e_indices = []
                for u, v in itertools.combinations(clq, 2):
                    e = (min(u, v), max(u, v))
                    if e in edge_to_idx:
                        e_indices.append(edge_to_idx[e])
                if len(e_indices) == 10:
                    # Mean pool edge embeddings of the clique
                    clq_e_emb = h_e[e_indices].mean(dim=0)
                    clique_edge_tensors.append(clq_e_emb)
                    valid_cliques.append(clq)

            if clique_edge_tensors:
                clq_stack = torch.stack(clique_edge_tensors)
                clq_scores = self.clique_head(clq_stack).squeeze(-1)
                # Map to multipliers
                for clq, score in zip(valid_cliques, clq_scores):
                    k5_tensor_multipliers[clq] = score
                    val = float(score.item())
                    k5_multipliers[clq] = val
                    candidate_rhs += val * 6.0
                    for u, v in itertools.combinations(clq, 2):
                        e = (min(u, v), max(u, v))
                        candidate_row[edge_to_idx[e]] += val

        return {
            "k5_multipliers": k5_multipliers,
            "k5_tensor_multipliers": k5_tensor_multipliers,
            "candidate_row": candidate_row,
            "candidate_rhs": float(candidate_rhs),
            "edge_multipliers": edge_mults.detach().cpu().numpy(),
        }


def train_surrogate_predictor(
    model: EdgeEquivariantSurrogateGNN,
    train_graphs: Sequence[GraphInstance],
    target_multipliers: Sequence[Dict[Tuple[int, int, int, int, int], float]],
    epochs: int = 25,
    lr: float = 0.01,
) -> List[float]:
    """Supervised training routine aligning predicted surrogate multipliers with canonical dual targets."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_history = []

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for graph, tgt_duals in zip(train_graphs, target_multipliers):
            optimizer.zero_grad()
            out = model(graph)

            tensor_k5 = out.get("k5_tensor_multipliers", {})
            if tensor_k5:
                pred_list = []
                true_list = []
                for clq, score_t in tensor_k5.items():
                    pred_list.append(score_t)
                    true_list.append(float(tgt_duals.get(clq, 0.0)))

                pred_t = torch.stack(pred_list)
                true_t = torch.tensor(true_list, dtype=torch.float32, device=pred_t.device)
                loss = F.mse_loss(pred_t, true_t)

                loss.backward()
                optimizer.step()
                ep_loss += float(loss.item())

        loss_history.append(ep_loss)

    model.eval()
    return loss_history
