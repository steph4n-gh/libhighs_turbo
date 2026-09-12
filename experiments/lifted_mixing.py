"""Research: low-rank moment fitting on an adaptively enlarged Boolean basis.

Repeated XOR products must have the same moment. Their multiplier sums are
zero, so they change the Gram representation without changing the polynomial.
The numerical fit proposes a factor; full coefficient repair checks its bound.
"""
from collections import defaultdict
from fractions import Fraction
from math import lcm
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, eigh
from scipy.optimize import minimize

from highs_turbo.ising_cuts import cut_matrix, make_certificate


def exact_gram_bound(coefficients, basis, integers, denominator):
    """Fast exact counterpart of the small reference polynomial checker."""
    n = len(basis)
    integer = np.asarray(integers)
    if (integer.ndim != 2 or integer.shape[0] != n or integer.dtype.kind not in 'iu'
            or type(denominator) is not int or denominator <= 0
            or len(set(basis)) != n or any(type(x) is not int or x < 0 for x in basis)):
        raise ValueError('Invalid polynomial Gram witness')
    maximum = max((abs(int(v)) for v in integer.ravel()),default=0)
    if integer.shape[1]*maximum**2 >= 2**53:
        raise ValueError('Integer dot products exceed the exact range')
    dense = integer.astype(float)
    gram = (dense @ dense.T).astype(np.int64)
    width = (max((m.bit_length() for m in [*basis,*coefficients]),default=0)+7)//8
    zero = bytes(width)
    expansion = defaultdict(int)
    for i, a in enumerate(basis):
        expansion[zero] += int(gram[i,i])
        for j in range(i):
            # Integer hashes reduce modulo 2**61-1: sparse spin masks then
            # collide heavily once there are hundreds of variables. Bytes
            # preserve exact identity and have a well-mixed hash.
            expansion[(a ^ basis[j]).to_bytes(width,'little')] += 2*int(gram[i,j])
    square = denominator**2
    numerator = -expansion[zero] - sum(abs(v) for m,v in expansion.items() if m != zero)
    result = coefficients.get(0,Fraction()) + Fraction(numerator,square)
    for mask, value in coefficients.items():
        if mask:
            remainder = Fraction(expansion.get(mask.to_bytes(width,'little'),0),square)
            result += abs(remainder)-abs(value-remainder)
    return result


def repeated_products(basis, base_count=0):
    """Store only repeated products; unique higher moments are free."""
    first, repeated = {}, {}
    width = (max((m.bit_length() for m in basis),default=0)+7)//8
    for i,a in enumerate(basis):
        if i < base_count:
            continue
        for j in range(i):
            mask = a ^ basis[j]
            key = mask.to_bytes(width,'little')
            if base_count and mask.bit_count() == 2 and key not in first:
                low = mask & -mask
                u, v = low.bit_length()-1, (mask ^ low).bit_length()-1
                first[key] = (v,u)
            if key in first:
                if key not in repeated:
                    repeated[key] = [first[key]]
                repeated[key].append((i,j))
            else:
                first[key] = (i,j)
    pairs, groups = [], []
    for group, occurrences in enumerate(repeated.values()):
        pairs.extend(occurrences)
        groups.extend([group]*len(occurrences))
    rows, cols = np.asarray(pairs,dtype=int).reshape(-1,2).T
    return rows, cols, np.asarray(groups,dtype=int), np.bincount(groups)


def lifted_mixing(n, edges, weights, pairs, *, cuts=(), seconds=5, seed=0, rank=24, penalty=1., initial_proof=None, triples=(), initial_vectors=None, equality_only=False):
    began = time.perf_counter()
    deadline = began+seconds
    phase_deadline = began+.75*seconds
    degree = np.bincount(np.asarray(edges).ravel(),minlength=n)
    anchor = int(degree.argmax())
    basis = [(1 << anchor) ^ (1 << i) for i in range(n)]
    seen = set(basis)
    for u,v in pairs:
        mask = (1 << int(u)) ^ (1 << int(v))
        if mask not in seen:
            seen.add(mask); basis.append(mask)
    for triple in triples:
        mask = (1 << anchor)
        for vertex in triple:
            mask ^= 1 << int(vertex)
        if mask not in seen:
            seen.add(mask); basis.append(mask)
    size = len(basis)
    rows, cols, groups, count = repeated_products(basis,n)
    u,v = np.asarray(edges).T
    original = np.asarray([float(weights[e]) for e in edges])
    cut_rows = cut_matrix(cuts,len(edges))
    shift = np.asarray(cut_rows.sum(axis=1)).ravel()-2*np.asarray([c.rhs for c in cuts])
    alpha, multipliers = np.zeros(len(rows)), np.zeros(len(cuts))
    if initial_proof is not None:
        known = {cut:float(Fraction(a,initial_proof.denominator))
                 for cut,a in zip(initial_proof.cuts,initial_proof.multipliers)}
        multipliers = np.array([known.get(c,0.) for c in cuts])
    rng = np.random.default_rng(seed)
    latest = rng.normal(size=(size,min(rank,size)))
    latest /= np.linalg.norm(latest,axis=1)[:,None]
    if initial_vectors is not None:
        rank = initial_vectors.shape[1]
        latest = np.empty((size, rank))
        latest[:n] = initial_vectors
        for index, mask in enumerate(basis[n:], n):
            triple = mask ^ (1 << anchor)
            nodes = [i for i in range(n) if (triple >> i) & 1]
            if len(nodes) != 3:
                raise ValueError('Warm start currently supports odd triples only')
            A = latest[nodes]
            rhs = np.array([A[1] @ A[2], A[0] @ A[2], A[0] @ A[1]])
            value = np.linalg.lstsq(A, rhs, rcond=1e-10)[0]
            size2 = value @ value
            if size2 > 1:
                value /= np.sqrt(size2)
            else:
                extra = rng.normal(size=rank)
                extra -= A.T @ np.linalg.lstsq(A @ A.T, A @ extra, rcond=1e-10)[0]
                norm = np.linalg.norm(extra)
                if norm > 1e-10:
                    value += extra * np.sqrt(max(0., 1-size2))/norm
            latest[index] = value
    latest = latest.ravel()
    rho, outer = 0., 0
    delta = np.zeros(len(rows))
    timings = {'setup':time.perf_counter()-began}

    def correlations(vectors):
        edge = np.einsum('ij,ij->i',vectors[u],vectors[v])
        product = np.einsum('ij,ij->i',vectors[rows],vectors[cols])
        delta = product-np.bincount(groups,weights=product,minlength=len(count))[groups]/count[groups]
        return edge,product,delta

    def objective(flat):
        vectors = flat.reshape(size,-1)
        norms = np.maximum(np.linalg.norm(vectors,axis=1)[:,None],1e-100)
        vectors = vectors/norms
        edge, product, delta = correlations(vectors)
        violation = shift-cut_rows @ edge
        active = np.maximum(0,multipliers+rho*violation) if rho else multipliers
        edge_values = original-cut_rows.T @ active
        product_values = alpha+rho*delta
        matrix = sp.csr_matrix((np.r_[edge_values,edge_values,product_values,product_values]/2,
                    (np.r_[u,v,rows,cols],np.r_[v,u,cols,rows])),shape=(size,size))
        applied = matrix @ vectors
        gradient = 2*(applied-vectors*np.sum(vectors*applied,axis=1)[:,None])/norms
        value = original @ edge + alpha @ product + rho/2*(delta @ delta)
        if rho:
            value += (active @ active-multipliers @ multipliers)/(2*rho)
        else:
            value += multipliers @ violation
        return value,gradient.ravel()

    def checkpoint(flat):
        nonlocal latest
        latest = flat.copy()
        if time.perf_counter() >= phase_deadline:
            raise StopIteration

    if equality_only and len(rows):
        calls = 0
        def equality_objective(beta):
            nonlocal alpha, latest, calls
            if calls and time.perf_counter() >= phase_deadline:
                raise StopIteration
            alpha = beta - np.bincount(groups, weights=beta, minlength=len(count))[groups]/count[groups]
            result = minimize(objective, latest, jac=True, method='L-BFGS-B', callback=checkpoint,
                              options={'maxiter':300 if calls==0 else 75, 'maxcor':5,
                                       'ftol':1e-11, 'gtol':1e-7})
            latest = result.x
            vectors = latest.reshape(size, -1)
            vectors = vectors/np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-100)
            _, _, difference = correlations(vectors)
            calls += 1
            return -objective(latest)[0], -difference
        try:
            result = minimize(equality_objective, alpha, jac=True, method='L-BFGS-B',
                              options={'maxiter':60,'maxls':10,'maxcor':5,'ftol':1e-10,'gtol':1e-7})
            alpha = result.x - np.bincount(groups, weights=result.x, minlength=len(count))[groups]/count[groups]
        except StopIteration:
            pass
        outer = calls
    while not equality_only and outer < 1000 and time.perf_counter() < phase_deadline:
        try:
            result = minimize(objective,latest,jac=True,method='L-BFGS-B',callback=checkpoint,
                              options={'maxiter':150 if outer==0 else 70,'maxcor':5,
                                       'ftol':1e-9,'gtol':1e-6,'maxls':15})
            latest = result.x
        except StopIteration:
            pass
        vectors = latest.reshape(size,-1)
        vectors /= np.maximum(np.linalg.norm(vectors,axis=1)[:,None],1e-100)
        latest = vectors.copy().ravel()
        edge,product,delta = correlations(vectors)
        alpha += rho*delta
        alpha -= np.bincount(groups,weights=alpha,minlength=len(count))[groups]/count[groups]
        multipliers = np.maximum(0,multipliers+rho*(shift-cut_rows @ edge))
        rho = penalty
        outer += 1
        if outer>3 and max(np.max(abs(delta),initial=0),np.max(shift-cut_rows @ edge,initial=0)) < 1e-5:
            break
    # Recover the ordinary SDP dual for these fixed equality multipliers.
    # Small equality violations alone do not imply a good dual certificate.
    rho = 0.
    phase_deadline = deadline
    try:
        result = minimize(objective,latest,jac=True,method='L-BFGS-B',callback=checkpoint,
                          options={'maxiter':400,'maxcor':5,'ftol':1e-11,'gtol':1e-7})
        latest = result.x
    except StopIteration:
        pass
    vectors = latest.reshape(size,-1)
    vectors /= np.maximum(np.linalg.norm(vectors,axis=1)[:,None],1e-100)
    timings['fit_done'] = time.perf_counter()-began
    proof = make_certificate(cuts,-multipliers,edges,weights,Fraction())
    residual = dict(weights)
    constant = Fraction()
    for cut, numerator in zip(proof.cuts,proof.multipliers):
        multiplier = Fraction(numerator,proof.denominator)
        constant += multiplier*(sum(cut.coefficients)-2*cut.rhs)
        for index, coefficient in zip(cut.indices,cut.coefficients):
            residual[edges[index]] -= multiplier*coefficient
    edge_values = np.asarray([float(residual[e]) for e in edges])
    matrix = sp.csr_matrix((np.r_[edge_values,edge_values,alpha,alpha]/2,
                (np.r_[u,v,rows,cols],np.r_[v,u,cols,rows])),shape=(size,size))
    diagonal = np.sum(vectors*(matrix @ vectors),axis=1)
    slack = matrix.toarray()
    slack.flat[::size+1] -= diagonal
    smallest = eigh(slack,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0]
    slack.flat[::size+1] += max(0.,-smallest)+1e-8
    # Only the requested triangle is part of the Cholesky factor. Explicitly
    # clear the other triangle before packing the general polynomial witness.
    factor = np.tril(cholesky(slack,lower=True,check_finite=False))
    largest = np.max(abs(factor))
    bits = min(20,int(np.floor(np.log2(np.sqrt(2**52/size)/largest))))
    denominator = 2**max(0,bits)
    integers = np.rint(factor*denominator).astype(np.int64)
    timings['factor_done'] = time.perf_counter()-began
    _, _, delta = correlations(vectors)
    coefficients = {(1 << int(u)) ^ (1 << int(v)):w for (u,v),w in residual.items()}
    coefficients[0] = constant
    exact = exact_gram_bound(coefficients,basis,integers,denominator)
    return dict(bound=float(exact),exact=exact,seconds=time.perf_counter()-began,
                bases=[size],repeated_terms=len(count),equalities=len(rows)-len(count),
                outer=outer,timings=timings,eigenvalue=float(smallest),violation=float(np.max(abs(delta),initial=0)),
                witness=(basis,integers,denominator),cut_proof=proof,moments=(vectors[:n] @ vectors[:n].T))
