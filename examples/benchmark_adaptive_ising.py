"""Held-out weighted/damaged Pegasus comparison, with exact gap checks.

Each method runs in a fresh worker. Timings include normalization, model
construction, search, and certificate checking; Python imports are excluded.
SMS is optional and its reported bounds remain explicitly numerical.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import highspy
import networkx as nx
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
from benchmark_ising import make_instance
from highs_turbo import solve_ising, verify_ising_certificate
from highs_turbo.ising import _normalize


def instance(size, seed, damaged):
    h, J = make_instance(8, size, seed, weighted=True)
    if damaged:
        rng = np.random.default_rng(seed+10000)
        h = {v: w for v, w in h.items() if rng.random() >= .05}
        J = {(u,v): w for (u,v), w in J.items() if u in h and v in h and rng.random() >= .1}
    return h, J


def exact_energy(h, J, spins):
    return sum((Fraction(w)*spins[v] for v,w in h.items()), Fraction()) + sum(
        (Fraction(w)*spins[u]*spins[v] for (u,v),w in J.items()), Fraction())


def worker(args):
    h, J = instance(args.size, args.seed, args.damaged)
    target = .2*(sum(abs(w) for w in h.values())+sum(abs(w) for w in J.values()))
    began = time.perf_counter()
    if args.worker in ('native', 'static', 'deterministic', 'learned'):
        result = solve_ising(h, J, time_limit=args.seconds, seed=args.repeat,
                             accelerate=args.worker != 'native', threads=1,
                             cut_policy=args.worker if args.worker != 'native' else 'deterministic')
        assert verify_ising_certificate(h, J, result.certificate)
        assert exact_energy(h, J, result.spins) == result.exact_energy
        elapsed = time.perf_counter()-began
        reached = [p['time'] for p in result.progress if p['energy']-p['lower_bound'] <= target]
        record = dict(seconds=elapsed, energy=result.energy, lower_bound=result.lower_bound,
                      numerical_gap=result.gap, verified_bound=float(result.exact_cut_lower_bound),
                      verified_gap=float(result.exact_gap), status=result.status, cuts=result.cuts_added,
                      subgraph_cuts=result.subgraph_cuts, rounds=result.root_rounds,
                      time_to_target=min(reached) if reached else None, progress=result.progress)
    elif args.worker == 'heuristic':
        deadline = began+args.seconds
        samples = SimulatedAnnealingSampler().sample_ising(
            h, J, num_reads=100000, num_sweeps=1000, seed=args.repeat,
            interrupt_function=lambda: time.perf_counter() >= began+args.seconds*.5)
        remaining = max(1, int(1000*(deadline-time.perf_counter())))
        tabu = TabuSampler().sample_ising(h, J, initial_states=samples.first.sample,
                                        num_reads=1, timeout=remaining, seed=args.repeat)
        spins = dict(min([samples.first, tabu.first], key=lambda sample: sample.energy).sample)
        energy = exact_energy(h, J, spins)
        record = dict(seconds=time.perf_counter()-began, energy=float(energy), lower_bound=None,
                      numerical_gap=None, verified_bound=None, verified_gap=None,
                      status='HEURISTIC', time_to_target=None)
    else:
        labels, fields, weights, constant = _normalize(h, J, 0)
        weights.update({(i, len(labels)): w for i,w in enumerate(fields) if w})
        with tempfile.TemporaryDirectory(prefix='ising-sms-') as directory:
            root = Path(directory)
            graph = root/'instance.mc'
            graph.write_text(f'{len(labels)+bool(any(fields))} {len(weights)}\n'+''.join(
                f'{u+1} {v+1} {float(w)}\n' for (u,v),w in sorted(weights.items())))
            command = [str(args.sms), str(graph), '--timelimit', str(int(args.seconds)),
                       '--seed', str(args.repeat), '--statsfile', str(root/'stats.json'),
                       '--solution', str(root/'solution.json')]
            output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=max(30, args.seconds*4))
            if output.returncode:
                raise RuntimeError(output.stdout[-4000:])
            stats = json.loads((root/'stats.json').read_text())
            partition = json.loads((root/'solution.json').read_text())
            bits = {i-1: 1 for i in partition['partition_0']}
            bits.update({i-1: -1 for i in partition['partition_1']})
            orientation = bits[len(labels)] if any(fields) else 1
            spins = {label: bits[i]*orientation for i,label in enumerate(labels)}
            energy = exact_energy(h, J, spins)
            full = stats['solver stats']
            cut_value = float(full['best solution value'])
            assert abs(float(constant+sum(weights.values()))-2*cut_value-float(energy)) < 1e-7
            remaining_gap = sum(max(0., row['final_bound_value']-row['best_solution_value'])
                                for key,row in full.items() if key.startswith('solver '))
            gap = 2*remaining_gap*stats['graph stats']['scaling factor']
            record = dict(seconds=time.perf_counter()-began, energy=float(energy),
                          lower_bound=float(energy)-gap, numerical_gap=gap, verified_bound=None,
                          verified_gap=None, status='OPTIMAL' if full['optimality proven'] else 'TIME_LIMIT',
                          time_to_target=None, solver_stats=full)
    record.update(method=args.worker, seed=args.seed, nodes=len(h), edges=sum(bool(w) for w in J.values()),
                  damaged=args.damaged, repeat=args.repeat, target_gap=target)
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--sizes', nargs='+', type=int, default=[0])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1001, 1002, 1003])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sms', type=Path)
    parser.add_argument('--json', type=Path, default=Path('adaptive-results.json'))
    parser.add_argument('--worker', choices=['native','static','deterministic','learned','heuristic','sms'])
    parser.add_argument('--size', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--damaged', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    methods = ['native', 'static', 'deterministic', 'learned', 'heuristic']+(['sms'] if args.sms else [])
    result = dict(platform=platform.platform(), python=platform.python_version(), highs=highspy.Highs().version(),
                  seconds=args.seconds, repeats=args.repeats, records=[])
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for size in args.sizes:
        for damaged in (False, True):
            for seed in args.seeds:
                for repeat in range(args.repeats):
                    order = np.roll(methods, repeat).tolist()
                    if (seed+repeat) % 2: order.reverse()
                    for method in order:
                        command = [sys.executable, __file__, '--worker', method, '--size', str(size),
                                   '--seed', str(seed), '--repeat', str(repeat), '--seconds', str(args.seconds)]
                        if damaged: command.append('--damaged')
                        if args.sms: command.extend(['--sms', str(args.sms)])
                        output = subprocess.run(command, env=env, text=True, capture_output=True,
                                                timeout=max(45, args.seconds*5), check=True)
                        row = json.loads(output.stdout)
                        result['records'].append(row)
                        args.json.write_text(json.dumps(result, indent=2)+'\n')
                        print(f"{row['nodes']:4} seed={seed} damaged={damaged} repeat={repeat} {method:13} "
                              f"E={row['energy']:9.3f} exactgap={str(row['verified_gap']):8} "
                              f"numgap={str(row['numerical_gap']):18} {row['seconds']:.3f}s", flush=True)
    # Check every numerical bound against the best independently evaluated
    # feasible state found by any method on the identical input.
    for row in result['records']:
        peers = [p for p in result['records'] if (p['nodes'],p['seed'],p['damaged']) ==
                 (row['nodes'],row['seed'],row['damaged'])]
        if row['lower_bound'] is not None:
            assert row['lower_bound'] <= min(p['energy'] for p in peers)+1e-6


if __name__ == '__main__':
    main()
