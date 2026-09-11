"""Compare complete cold solves on official Pegasus subgraphs.

python examples/benchmark_ising.py --sizes 32 48 64 --seeds 1 2 3 --repeats 3
python examples/benchmark_ising.py --sizes 128 256 512 --m 8 --time-limit 5
Size 0 selects the full fabric. No QPU connection or cached solve is used.
"""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import highspy
import networkx as nx
import numpy as np
from dwave.graphs import pegasus_graph
from highs_turbo import solve_ising


def make_instance(m, size, seed, weighted=False):
    topology = pegasus_graph(m)
    if size < 0 or size > len(topology):
        raise ValueError(f"size must be 0 through {len(topology)} for P_{m}")
    labels = list(topology)
    selected = list(nx.bfs_tree(topology, labels[(seed * 7) % len(labels)]))[:size or len(labels)]
    graph = topology.subgraph(selected)
    rng = np.random.default_rng(seed)
    J = {edge: float(rng.integers(-8, 9)) / 8 if weighted else int(rng.choice([-1, 1]))
         for edge in sorted(graph.edges)}
    h = {node: float(rng.integers(-3, 4)) / 8 if weighted else 0 for node in graph}
    return h, J


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--sizes', type=int, nargs='+', default=[32, 48, 64])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--time-limit', type=float, default=10)
    parser.add_argument('--weighted', action='store_true', help='Weighted couplings and nonzero local fields')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    records = []
    print(f'{platform.platform()} | Python {platform.python_version()} | HiGHS {highspy.Highs().version()}', flush=True)
    print('nodes seed | native ms / gap | turbo ms / gap | speedup (both optimal)', flush=True)
    for size in args.sizes:
        for seed in args.seeds:
            h, J = make_instance(args.m, size, seed, args.weighted)
            runs = {'native': [], 'turbo': []}
            for repeat in range(args.repeats):
                order = ['native', 'turbo'] if repeat % 2 == 0 else ['turbo', 'native']
                for method in order:
                    result = solve_ising(h, J, time_limit=args.time_limit, seed=0, accelerate=method == 'turbo')
                    original = sum(w * result.spins[i] for i, w in h.items()) + sum(w * result.spins[i] * result.spins[j] for (i, j), w in J.items())
                    assert set(result.spins) == set(h) and set(result.spins.values()) <= {-1, 1}
                    assert abs(original - result.energy) <= 1e-8
                    assert result.lower_bound <= result.energy + 1e-7
                    assert result.status != 'SOLVER_ERROR', result.message
                    runs[method].append({'time_ms': result.solve_time * 1000, 'energy': result.energy,
                                         'lower_bound': result.lower_bound, 'gap': result.gap,
                                         'status': result.status, 'nodes': result.node_count,
                                         'cuts': result.cuts_added, 'cut_lower_bound': result.cut_lower_bound})
            all_runs = runs['native'] + runs['turbo']
            best_energy = min(run['energy'] for run in all_runs)
            assert max(run['lower_bound'] for run in all_runs) <= best_energy + 1e-6
            optimal = [run for run in all_runs if run['status'] == 'OPTIMAL']
            assert all(abs(run['energy'] - best_energy) <= 1e-6 for run in optimal)
            medians = {method: statistics.median(run['time_ms'] for run in samples) for method, samples in runs.items()}
            gaps = {method: statistics.median(run['gap'] for run in samples) for method, samples in runs.items()}
            both_optimal = len(optimal) == len(all_runs)
            speedup = medians['native'] / medians['turbo'] if both_optimal else None
            records.append({'m': args.m, 'nodes': len(h), 'edges': len(J), 'seed': seed,
                            'weighted': args.weighted, 'runs': runs, 'median_ms': medians,
                            'median_gap': gaps, 'speedup': speedup})
            print(f"{len(h):5} {seed:4} | {medians['native']:9.2f} / {gaps['native']:7.2f} | "
                  f"{medians['turbo']:9.2f} / {gaps['turbo']:7.2f} | "
                  + (f'{speedup:.2f}x' if speedup else 'time-limited; compare energy and gap'), flush=True)
    if args.json:
        args.json.write_text(json.dumps({'platform': platform.platform(), 'python': platform.python_version(),
                                        'highs': highspy.Highs().version(), 'time_limit': args.time_limit,
                                        'repeats': args.repeats, 'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
