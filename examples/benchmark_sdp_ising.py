"""Compare global and cut bounds on fresh full-size weighted Pegasus graphs.

Six fixed-budget cases, then three predetermined time-to-gap comparisons.
Each solve uses a fresh process and includes independent certificate checking.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_adaptive_ising import instance, exact_energy
from highs_turbo import solve_ising, verify_ising_certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['cuts', 'sdp', 'hybrid'])
    parser.add_argument('--seed', type=int, default=3001)
    parser.add_argument('--damaged', action='store_true')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--json', type=Path, default=Path('sdp-results.json'))
    parser.add_argument('--witness', type=Path)
    args = parser.parse_args()
    if args.worker:
        h, J = instance(0, args.seed, args.damaged)
        target = .075*(sum(abs(w) for w in h.values())+sum(abs(w) for w in J.values()))
        began = time.perf_counter()
        result = solve_ising(h, J, time_limit=args.seconds, relaxation=args.worker, threads=1,
                             certified_gap=target if args.target else None)
        assert verify_ising_certificate(h, J, result.certificate)
        assert exact_energy(h, J, result.spins) == result.exact_energy
        elapsed = time.perf_counter()-began
        record = dict(method=args.worker, seed=args.seed, damaged=args.damaged, nodes=len(h),
                      budget=args.seconds, target=target if args.target else None, seconds=elapsed,
                      energy=result.energy, bound=float(result.exact_cut_lower_bound),
                      gap=float(result.exact_gap), status=result.status, progress=result.progress,
                      gram_vertices=len(result.certificate.gram_factor))
        if args.witness:
            with gzip.open(args.witness, 'wt') as stream:
                json.dump(result.certificate.to_dict(), stream, separators=(',', ':'))
        print(json.dumps(record))
        return
    cases = [(seed, damaged) for damaged in (False, True) for seed in (3001, 3002, 3003)]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    output = dict(target_fraction=.075, cases=cases, records=[])
    jobs = [(seed, damaged, method, 5, False) for i, (seed, damaged) in enumerate(cases)
            for method in (['cuts', 'sdp', 'hybrid'] if i % 2 == 0 else ['hybrid', 'sdp', 'cuts'])]
    # Chosen before outcomes: first/third intact cases and second damaged case.
    jobs += [(seed, damaged, method, budget, True) for seed, damaged in cases[::2]
             for method, budget in [('hybrid', 5), ('cuts', 45)]]
    for seed, damaged, method, budget, target in jobs:
        command = [sys.executable, __file__, '--worker', method, '--seed', str(seed),
                   '--seconds', str(budget)]
        if damaged:
            command.append('--damaged')
        if target:
            command.append('--target')
        if not target and seed == 3001 and not damaged and method == 'hybrid' and args.witness:
            command.extend(['--witness', str(args.witness)])
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=budget+60)
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        record = json.loads(completed.stdout)
        output['records'].append(record)
        args.json.write_text(json.dumps(output, indent=2)+'\n')
        print(f'{seed} damaged={damaged} {method} target={target} '
              f'gap={record["gap"]:.3f} time={record["seconds"]:.3f}s', flush=True)


if __name__ == '__main__':
    main()
