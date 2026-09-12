"""Research prototype: selected Boolean monomials and exactly checked SOS bounds.

Optional research dependency: cvxpy. No production API depends on this module.
Monomials are integer bit masks; multiplication is XOR because every spin
squares to one. All expanded coefficients, including degree-four fill-in,
are retained by the checker.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh


def polynomial(n, edges, weights):
    result = {0: Fraction(0)}
    for (u, v), w in zip(edges, weights):
        mask = (1 << int(u)) ^ (1 << int(v))
        result[mask] = result.get(mask, Fraction(0)) + Fraction(float(w))
    return result


def check_sos(coefficients, blocks):
    """Return an exact lower bound from arbitrary rational Gram factors.

    Each block is (monomial masks, integer factor, positive denominator).
    This simple reference implementation uses Python integers throughout.
    """
    residual = defaultdict(Fraction, coefficients)
    for basis, factor, denominator in blocks:
        if type(denominator) is not int or denominator <= 0:
            raise ValueError('Expected positive integer denominator')
        if len(basis) != len(factor) or len(set(basis)) != len(basis):
            raise ValueError('Invalid monomial basis')
        if any(type(m) is not int or m < 0 for m in basis):
            raise ValueError('Expected nonnegative integer monomial masks')
        rank = len(factor[0]) if factor else 0
        if any(len(row) != rank or any(type(x) is not int for x in row) for row in factor):
            raise ValueError('Expected rectangular integer factor')
        square = denominator * denominator
        for i, a in enumerate(basis):
            for j in range(i, len(basis)):
                value = sum(x*y for x, y in zip(factor[i], factor[j]))
                residual[a ^ basis[j]] -= Fraction(value * (1 if i == j else 2), square)
    return residual[0] - sum(abs(v) for m, v in residual.items() if m)


def quantize_gram(basis, gram, bits=25):
    # Projection/factorization only constructs a proposal. check_sos is sound
    # for every integer factor, whether or not the numerical solve converged.
    eig, vec = np.linalg.eigh((gram + gram.T)/2)
    factor = vec * np.sqrt(np.maximum(eig, 0))
    denominator = 2**bits
    integers = [[int(x) for x in row] for row in np.rint(factor*denominator)]
    return list(basis), integers, denominator


def solve_sos(coefficients, bases, *, seconds=20, solver='CLARABEL'):
    """Solve the chosen SOS formulation and return an exact repaired witness."""
    began = time.perf_counter()
    support = sorted(set(coefficients) | {a ^ b for basis in bases for a in basis for b in basis})
    location = {m: i for i, m in enumerate(support)}
    rhs = np.array([float(coefficients.get(m, 0)) for m in support])
    terms, grams = [], []
    for basis in bases:
        k = len(basis)
        gram = cp.Variable((k, k), PSD=True)
        grams.append(gram)
        rows = [location[a ^ b] for b in basis for a in basis]
        operator = sp.csc_matrix((np.ones(k*k), (rows, np.arange(k*k))), shape=(len(support), k*k))
        terms.append(operator @ cp.vec(gram, order='F'))
    bound = cp.Variable()
    constant = np.zeros(len(support)); constant[location[0]] = 1
    match = sum(terms) + constant*bound == rhs
    problem = cp.Problem(cp.Maximize(bound), [match])
    if solver == 'CLARABEL':
        problem.solve(solver=solver, time_limit=seconds, max_iter=200,
                      tol_gap_abs=1e-7, tol_feas=1e-7, tol_gap_rel=1e-7)
    else:
        problem.solve(solver=solver, time_limit_secs=seconds, eps=1e-5)
    if any(g.value is None for g in grams):
        raise RuntimeError(f'SOS solver returned {problem.status}')
    witnesses = [quantize_gram(basis, g.value) for basis, g in zip(bases, grams)]
    exact = check_sos(coefficients, witnesses)
    return dict(bound=float(exact), exact=exact, numerical=problem.value,
                seconds=time.perf_counter()-began, status=problem.status,
                bases=[len(b) for b in bases], support=len(support),
                witness=witnesses, moments=dict(zip(support, match.dual_value)))


def singleton_basis(n):
    return [1 << i for i in range(n)]


def pair_basis(edges):
    return [0] + sorted({(1 << int(u)) ^ (1 << int(v)) for u, v in edges})


def anchored_basis(n, pairs, anchor=0):
    """Fix the global spin-flip symmetry before adding selected products.

    Multiplying all basis monomials by the anchor spin leaves every square
    unchanged. The base rows become 1 and s_anchor*s_i. Adding s_u*s_v
    therefore couples the new products directly to the original relaxation.
    """
    return sorted({(1 << anchor) ^ (1 << i) for i in range(n)}
                  | {(1 << int(u)) ^ (1 << int(v)) for u, v in pairs})


def solve_sos_admm(coefficients, bases, *, seconds=10, iterations=2000):
    """Direct Gram search with a closed-form coefficient-residual step.

    The objective charges the full absolute residual, exactly as the checker
    does. An interrupted solve still has a valid certificate. Dense PSD
    projections are appropriate only for bounded research blocks.
    """
    began = time.perf_counter()
    masks = sorted({a ^ b for basis in bases for a in basis for b in basis if a != b})
    locations = {m: i for i, m in enumerate(masks)}
    target = np.array([float(coefficients.get(m, 0)) for m in masks])
    missing = sum(abs(float(v)) for m, v in coefficients.items() if m and m not in locations)
    indices, groups = [], []
    for basis in bases:
        row, col = np.triu_indices(len(basis), 1)
        indices.append((row, col))
        groups.append(np.array([locations[basis[i] ^ basis[j]] for i, j in zip(row, col)]))
    count = sum(np.bincount(g, minlength=len(masks)) for g in groups)
    z = [np.zeros((len(b),len(b))) for b in bases]
    dual = [v.copy() for v in z]
    rho, best, best_x, best_eigen = 1., -float('inf'), None, None
    for iteration in range(iterations):
        x, eigens = [], []
        for zv, uv in zip(z, dual):
            eig, vec = eigh(zv-uv, check_finite=False, driver='evr')
            eig = np.maximum(eig, 0)
            factor = vec * np.sqrt(eig)
            x.append(factor @ factor.T)
            eigens.append(factor)
        expansion = sum(2*np.bincount(g, weights=v[i,j], minlength=len(masks))
                        for v, (i,j), g in zip(x, indices, groups))
        value = float(coefficients.get(0,0))-sum(np.trace(v) for v in x)-sum(abs(target-expansion))-missing
        if value > best:
            best, best_x, best_eigen = value, [v.copy() for v in x], eigens
        v = [a+b for a,b in zip(x,dual)]
        expansion = sum(2*np.bincount(g, weights=a[i,j], minlength=len(masks))
                        for a, (i,j), g in zip(v, indices, groups))
        adjustment = np.clip((target-expansion)/(2*count), -1/rho, 1/rho)
        new_z = []
        for a, (i,j), g in zip(v,indices,groups):
            a[i,j] += adjustment[g]
            a[j,i] = a[i,j]
            a.flat[::len(a)+1] -= 1/rho
            new_z.append(a)
        primal = np.sqrt(sum(np.sum((a-b)**2) for a,b in zip(x,new_z)))
        dual_residual = rho*np.sqrt(sum(np.sum((a-b)**2) for a,b in zip(z,new_z)))
        dual = [u+a-b for u,a,b in zip(dual,x,new_z)]
        z = new_z
        if iteration % 25 == 24:
            change = 2. if primal > 5*dual_residual else .5 if dual_residual > 5*primal else 1.
            rho *= change
            dual = [u/change for u in dual]
        if time.perf_counter()-began >= seconds or max(primal,dual_residual) < 1e-5:
            break
    denominator = 2**25
    witnesses = [(list(basis), [[int(v) for v in row] for row in np.rint(factor*denominator)], denominator)
                 for basis,factor in zip(bases,best_eigen)]
    exact = check_sos(coefficients,witnesses)
    # At convergence the scaled dual gives the pseudo-moments used to choose
    # the next products. Repeated occurrences are averaged.
    moments = sum(np.bincount(g,weights=rho*u[i,j],minlength=len(masks))
                  for u,(i,j),g in zip(dual,indices,groups))/count
    return dict(bound=float(exact), exact=exact, numerical=best,
                seconds=time.perf_counter()-began, iterations=iteration+1,
                bases=[len(b) for b in bases], support=len(masks)+1,
                witness=witnesses, moments=dict(zip(masks,moments)))


def selected_pairs(n, edges, moments, budget, method='frustrated'):
    """Seed a sparse lift with wedges the current moments cannot realize.

    For each triangle, the four possible pair assignments have nonnegative
    probabilities. Their most negative pseudo-probability ranks the wedge.
    This is an initial discovery heuristic, not a novelty claim.
    """
    correlation = np.eye(n)
    for i, j in combinations(range(n), 2):
        correlation[i, j] = correlation[j, i] = moments.get((1 << i) ^ (1 << j), 0)
    candidates = []
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        adjacency[u].add(v); adjacency[v].add(u)
    for j in range(n):
        for i, k in combinations(sorted(adjacency[j]), 2):
            a, b, c = correlation[i,j], correlation[j,k], correlation[i,k]
            violation = -min(1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c)
            candidates.append((violation, tuple(sorted((i,j,k)))))
    chosen = set()
    for score, vertices in sorted(set(candidates), reverse=True):
        additions = {tuple(sorted(pair)) for pair in combinations(vertices, 2)} - chosen
        if len(chosen) + len(additions) <= budget:
            chosen.update(additions)
    return sorted(chosen)
