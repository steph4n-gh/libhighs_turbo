"""Automated tests for surrogate GNN model and topological features."""

import itertools
import pytest
import torch
import numpy as np

from highs_turbo.graph_generator import (
    generate_ab_pair,
    generate_k5_cluster_graph,
)
from highs_turbo.surrogate_model import (
    EdgeEquivariantSurrogateGNN,
    extract_topological_features,
    train_surrogate_predictor,
)


def test_topological_features_harmonic_rarity():
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    node_feats, edge_feats, edge_to_idx = extract_topological_features(g)

    assert node_feats.shape[0] == g.num_nodes
    assert edge_feats.shape[0] == g.num_edges

    # Check edge features include harmonic rarity (col 1) and harmonic degree (col 0)
    for i in range(g.num_edges):
        harm_deg = edge_feats[i, 0].item()
        harm_rarity = edge_feats[i, 1].item()
        assert harm_deg >= 0.0
        assert harm_rarity >= 0.0
        # Rarity <= Harmonic degree
        assert harm_rarity <= harm_deg + 1e-6


def test_surrogate_model_permutation_equivariance():
    g = generate_k5_cluster_graph(num_k5=3, num_bridges=2, seed=99)
    ga, gb, pi, pi_inv = generate_ab_pair(g, seed=123)

    model = EdgeEquivariantSurrogateGNN(hidden_dim=16, num_layers=2)
    model.eval()

    with torch.no_grad():
        out_a = model(ga)
        out_b = model(gb)

    # Candidate RHS must match across A and B
    assert abs(out_a["candidate_rhs"] - out_b["candidate_rhs"]) < 1e-4

    # Multipliers on mapped cliques must match
    mults_a = out_a["k5_multipliers"]
    mults_b = out_b["k5_multipliers"]
    assert len(mults_a) == len(mults_b)

    for clq_a, val_a in mults_a.items():
        clq_b = tuple(sorted([pi[v] for v in clq_a]))
        assert clq_b in mults_b
        val_b = mults_b[clq_b]
        assert abs(val_a - val_b) < 1e-4


def test_surrogate_model_training():
    g = generate_k5_cluster_graph(num_k5=2, num_bridges=1, seed=42)
    model = EdgeEquivariantSurrogateGNN(hidden_dim=16, num_layers=2)

    cliques = g.find_all_k5_cliques()
    targets = [{clq: 1.0 for clq in cliques}]

    weights_before = [p.clone() for p in model.parameters()]
    loss_hist = train_surrogate_predictor(model, [g], targets, epochs=15, lr=0.03)

    assert len(loss_hist) == 15
    # Parameters must be updated via autograd
    weights_after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(weights_before, weights_after))

    # Loss must strictly decrease
    assert loss_hist[-1] < loss_hist[0]
    assert loss_hist[-1] < 0.01

    # Evaluation predictions must approach targets
    with torch.no_grad():
        pred = model(g)
    for clq in cliques:
        assert abs(pred["k5_multipliers"][clq] - 1.0) < 0.15
