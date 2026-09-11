"""Solve a genuine Pegasus subgraph locally; no QPU credentials are needed."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
from dwave.graphs import pegasus_graph
from highs_turbo import solve_ising
from highs_turbo.topologies import generate_pegasus_instance


def run_dwave_pegasus_demo():
    fabric = pegasus_graph(4)
    nodes = list(nx.bfs_tree(fabric, next(iter(fabric))))[:32]
    graph = generate_pegasus_instance(4, seed=42, node_list=nodes)
    labels = graph.metadata['node_labels']
    J = {(labels[u], labels[v]): weight for (u, v), weight in graph.weights.items()}
    result = solve_ising({label: 0 for label in labels}, J, time_limit=10)
    print(f'Pegasus P_4 subgraph: {len(labels)} spins, {len(J)} couplers')
    print(f'Status: {result.status}; energy: {result.energy}; lower bound: {result.lower_bound}')
    print(f'Absolute gap: {result.gap:g}; elapsed: {result.solve_time:.3f} seconds')
    print(f'Cycle cuts: {result.cuts_added}; exact rational optimality: {result.is_rationally_certified}')
    print(f'Spins in the original D-Wave labels: {result.spins}')
    assert result.energy == sum(w * result.spins[u] * result.spins[v] for (u, v), w in J.items())


if __name__ == '__main__':
    run_dwave_pegasus_demo()
