"""Train the optional cut ranking model from measured marginal LP gains.

Development instances use seeds 101 onward; the benchmark holds out 1001
onward. Candidate labels include separator time and actual root-bound gain.
Run with the ``ml`` extra installed. Production inference only uses NumPy.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networkx as nx
import numpy as np
import torch
from benchmark_ising import make_instance
from highs_turbo.ising import _adaptive_relaxation, _normalize
from highs_turbo.ising_policy import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=int, default=24)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--data', type=Path, default=Path('/tmp/ising-policy-training.json'))
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1]/'highs_turbo/ising_policy.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    records = []
    for index in range(args.instances):
        seed = 101+index
        size = [32, 64, 128, 256][index % 4]
        h, J = make_instance(8, size, seed, weighted=True)
        if index % 2:
            rng = np.random.default_rng(seed+10000)
            J = {edge: w for edge, w in J.items() if rng.random() >= .1}
        labels, fields, weights, constant = _normalize(h, J, 0)
        weights.update({(i, len(labels)): w for i, w in enumerate(fields) if w})
        edges = sorted(weights)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(labels)+bool(any(fields))))
        graph.add_edges_from(edges)
        samples = []
        started = time.perf_counter()
        _adaptive_relaxation(graph, edges, weights, constant, Fraction(10**9),
                             deadline=started+args.seconds, started=started, seed=seed,
                             policy='deterministic', threads=1, training_samples=samples)
        for sample in samples:
            sample.update(seed=seed, size=size)
        records.extend(samples)
        print(f'seed {seed}: {len(samples)} candidates, {sum((s["bound_gain"] or 0)>1e-8 for s in samples)} positive', flush=True)
    args.data.write_text(json.dumps(records, indent=2)+'\n')
    usable = [s for s in records if s['bound_gain'] is not None]
    if len(usable) < 40:
        raise RuntimeError('Too few measured labels for a ranking model')
    x = np.asarray([s['features'] for s in usable], dtype=np.float32)
    target = np.asarray([np.log1p(s['bound_gain']/max(s['seconds'], .001)) for s in usable], dtype=np.float32)
    validation = np.asarray([(s['seed']-101) % 6 == 5 for s in usable])
    if not validation.any() or validation.all():
        raise RuntimeError('Need independent training and validation instances')
    mean, scale = x[~validation].mean(axis=0), np.maximum(x[~validation].std(axis=0), .01)
    xx, yy = torch.from_numpy((x-mean)/scale), torch.from_numpy(target[:, None])
    network = torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    optimizer = torch.optim.Adam(network.parameters(), lr=.01, weight_decay=.01)
    train = torch.from_numpy(~validation)
    for _ in range(400):
        optimizer.zero_grad()
        loss = torch.mean((network(xx[train])-yy[train])**2)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = network(xx).numpy().ravel()
    val_mse = float(np.mean((prediction[validation]-target[validation])**2))
    mean_mse = float(np.mean((target[~validation].mean()-target[validation])**2))
    result = dict(version=1, features=list(FEATURE_NAMES), mean=mean.tolist(), scale=scale.tolist(),
                  w1=network[0].weight.detach().numpy().T.tolist(), b1=network[0].bias.detach().numpy().tolist(),
                  w2=network[2].weight.detach().numpy()[0].tolist(), b2=float(network[2].bias.detach()[0]),
                  training=dict(seeds=list(range(101, 101+args.instances)), candidates=len(usable),
                                validation_seeds=sorted({s['seed'] for s, v in zip(usable, validation) if v}),
                                validation_mse=val_mse, constant_predictor_mse=mean_mse,
                                target='log1p(marginal LP energy-bound gain / max(separator seconds, 0.001))'))
    args.model.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['training'], indent=2))


if __name__ == '__main__':
    main()
