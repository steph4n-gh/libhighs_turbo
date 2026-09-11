"""Research: discover nonlocal integer-parity relations in low-rank geometry.

For integer b with odd coefficient sum, (b.s)**2 >= 1 for every spin state.
Nearest-neighbor queries find short odd sums of the current vectors; they are
only proposals. Parity, multipliers, and the final residual are checked exactly.
"""
from fractions import Fraction
from itertools import combinations
from math import lcm
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky,eigh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from highs_turbo.ising_cuts import make_certificate,cut_matrix
from highs_turbo.ising_sdp import gram_lower_bound


def discover_parity(vectors, edges, *, limit=512, seed=0, depth=3, minimum_support=3):
    began = time.perf_counter()
    n,rank = vectors.shape
    signed = np.r_[vectors,-vectors]
    tree = cKDTree(signed)
    rng = np.random.default_rng(seed)
    candidates = {}

    def offer(coefficients):
        coefficients = {int(i):int(b) for i,b in coefficients.items() if b}
        if not coefficients or sum(coefficients.values())%2 != 1:
            return
        ordered = sorted(coefficients.items())
        if ordered[0][1]<0:
            ordered = [(i,-b) for i,b in ordered]
        key = tuple(ordered)
        vector = sum(b*vectors[i] for i,b in key)
        norm = float(vector @ vector)
        if norm < .99:
            candidates[key] = (norm,vector)

    first = np.r_[np.asarray(edges),rng.integers(0,n,size=(8*n,2))]
    for sign in (1,-1):
        target = -(vectors[first[:,0]]+sign*vectors[first[:,1]])
        _,neighbors = tree.query(target,k=3,workers=1)
        for (u,v),near in zip(first,neighbors):
            if u==v:
                continue
            for signed_index in near:
                w = int(signed_index)%n
                if w not in (u,v):
                    offer({int(u):1,int(v):sign,w:1 if signed_index<n else -1})
                    break
    for level in range(depth):
        seeds = sorted(candidates,key=lambda b:candidates[b][0])[:min(limit,128)]
        for key in seeds:
            residual = candidates[key][1]
            indices = rng.integers(0,2*n,size=min(128,2*n))
            target = -(residual+signed[indices])
            _,neighbors = tree.query(target,k=1,workers=1)
            for i,j in zip(indices,neighbors):
                proposal = dict(key)
                for index in (int(i),int(j)):
                    node = index%n
                    proposal[node] = proposal.get(node,0)+(1 if index<n else -1)
                offer(proposal)
    ordered = sorted((b for b in candidates if len(b)>=minimum_support),
                     key=lambda b:-(1-candidates[b][0])/sum(a*a for i,a in b))
    chosen,usage = [],np.zeros(n,dtype=int)
    for cut in ordered:
        if any(usage[i]>=16 for i,a in cut):
            continue
        chosen.append(cut)
        for i,a in cut:usage[i]+=1
        if len(chosen)>=limit:break
    return chosen,dict(seconds=time.perf_counter()-began,candidates=len(candidates),
                       maximum_support=max(map(len,chosen),default=0),
                       strongest_violation=1-min((candidates[b][0] for b in chosen),default=1))


def parity_matrix(cuts,n):
    rows,cols,values=[],[],[]
    for row,cut in enumerate(cuts):
        for node,coefficient in cut:
            rows.append(row);cols.append(node);values.append(coefficient)
    return sp.csr_matrix((values,(rows,cols)),shape=(len(cuts),n),dtype=float)


def parity_bound(edges,weights,source,parity_cuts,multipliers,factor,denominator):
    """Check arbitrary-support parity cuts and all Gram fill-in exactly."""
    residual = dict(weights)
    constant = Fraction()
    for cut,a in zip(source.cuts,source.multipliers):
        value = Fraction(a,source.denominator)
        constant += value*(sum(cut.coefficients)-2*cut.rhs)
        for i,b in zip(cut.indices,cut.coefficients):
            residual[edges[i]] -= value*b
    for cut,a in zip(parity_cuts,multipliers):
        if (type(a) is not int or a<0 or len(set(i for i,b in cut))!=len(cut)
                or any(type(i) is not int or type(b) is not int or b==0 for i,b in cut)
                or sum(b for i,b in cut)%2!=1):
            raise ValueError('Invalid integer-parity witness')
        value = Fraction(a,2**20)
        constant += value*(1-sum(b*b for i,b in cut))
        for (u,b),(v,c) in combinations(cut,2):
            edge = tuple(sorted((u,v)))
            residual[edge] = residual.get(edge,Fraction())-2*value*b*c
    expanded = sorted(residual)
    scale = lcm(*(w.denominator for w in residual.values()))
    integers = [w.numerator*(scale//w.denominator) for e in expanded for w in (residual[e],)]
    return constant+gram_lower_bound(expanded,integers,scale,factor,denominator)


def fit_parity(n,edges,weights,source,parity_cuts,initial_vectors,*,seconds=5,joint=False,initial_multipliers=None):
    began = time.perf_counter();deadline=began+seconds;phase_deadline=began+.8*seconds
    row = cut_matrix(source.cuts,len(edges))
    shift = np.asarray(row.sum(axis=1)).ravel()-2*np.array([c.rhs for c in source.cuts])
    original = np.array([float(weights[e]) for e in edges]);u,v=np.asarray(edges).T
    fixed = np.array([float(Fraction(a,source.denominator)) for a in source.multipliers])
    B = parity_matrix(parity_cuts,n)
    latest = initial_vectors.copy().ravel()
    initial = (np.zeros(len(parity_cuts)) if initial_multipliers is None
               else np.asarray(initial_multipliers)/2**20)
    chosen = np.r_[fixed,initial] if joint else initial
    calls = 0

    def split(multipliers):
        return (multipliers[:len(fixed)],multipliers[len(fixed):]) if joint else (fixed,multipliers)

    def fit(multipliers,iterations):
        nonlocal latest
        edge_lambda,parity_lambda = split(multipliers)
        residual = original-row.T @ edge_lambda
        C = sp.csr_matrix((np.r_[residual,residual]/2,(np.r_[u,v],np.r_[v,u])),shape=(n,n))
        def objective(flat):
            V = flat.reshape(n,-1);norms=np.maximum(np.linalg.norm(V,axis=1)[:,None],1e-100);V=V/norms
            BV = B@V
            applied = C@V-B.T@(parity_lambda[:,None]*BV)
            diagonal = np.sum(V*applied,axis=1)
            return sum(diagonal),(2*(applied-V*diagonal[:,None])/norms).ravel()
        def checkpoint(flat):
            nonlocal latest
            latest=flat.copy()
            if time.perf_counter()>=phase_deadline:raise StopIteration
        try:
            result=minimize(objective,latest,jac=True,method='L-BFGS-B',callback=checkpoint,
                            options={'maxiter':iterations,'maxcor':5,'ftol':1e-10,'gtol':1e-7})
            latest=result.x
        except StopIteration:pass
        return objective(latest)[0]+edge_lambda@shift+sum(parity_lambda)

    def outer(multipliers):
        nonlocal chosen,calls
        if calls and time.perf_counter()>=phase_deadline:raise StopIteration
        chosen=multipliers.copy()
        value=fit(chosen,300 if calls==0 else 150);calls+=1
        V=latest.reshape(n,-1);V=V/np.linalg.norm(V,axis=1)[:,None]
        parity_gradient=np.sum((B@V)**2,axis=1)-1
        gradient=np.r_[row@np.einsum('ij,ij->i',V[u],V[v])-shift,parity_gradient] if joint else parity_gradient
        return -value,gradient
    try:
        result=minimize(outer,chosen,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(chosen),
                        options={'maxiter':60,'maxls':10,'maxcor':5,'ftol':1e-9,'gtol':1e-6})
        chosen=result.x
    except StopIteration:pass
    edge_lambda,parity_lambda=split(chosen)
    source=make_certificate(source.cuts,-edge_lambda,edges,weights,Fraction())
    parity_integer=[max(0,round(float(a)*2**20)) for a in parity_lambda]
    parity_lambda=np.array(parity_integer)/2**20
    fixed=np.array([float(Fraction(a,source.denominator)) for a in source.multipliers])
    row=cut_matrix(source.cuts,len(edges))
    shift=np.asarray(row.sum(axis=1)).ravel()-2*np.array([c.rhs for c in source.cuts])
    chosen=np.r_[fixed,parity_lambda] if joint else parity_lambda
    phase_deadline=deadline
    fit(chosen,400)
    V=latest.reshape(n,-1);V/=np.linalg.norm(V,axis=1)[:,None]
    residual=original-row.T@fixed
    C=sp.csr_matrix((np.r_[residual,residual]/2,(np.r_[u,v],np.r_[v,u])),shape=(n,n))-B.T@sp.diags(parity_lambda)@B
    slack=C.toarray();slack.flat[::n+1]-=np.sum(V*(C@V),axis=1)
    smallest=eigh(slack,subset_by_index=[0,0],eigvals_only=True,check_finite=False)[0]
    slack.flat[::n+1]+=max(0.,-smallest)+1e-8
    factor=np.tril(cholesky(slack,lower=True,check_finite=False));largest=abs(factor).max()
    bits=min(20,int(np.floor(np.log2(np.sqrt(2**52/n)/largest))));denominator=2**max(0,bits)
    integers=np.rint(factor*denominator).astype(np.int64)
    packed=tuple(tuple(map(int,integers[i,:i+1])) for i in range(n))
    lower=parity_bound(edges,weights,source,parity_cuts,parity_integer,packed,denominator)
    return dict(bound=float(lower),exact=lower,seconds=time.perf_counter()-began,calls=calls,
                active=sum(a>0 for a in parity_integer),eigenvalue=float(smallest),vectors=V,
                source=source,parity_cuts=parity_cuts,multipliers=parity_integer,factor=packed,denominator=denominator)


def fit_augmented(n,edges,weights,source,parity_cuts,initial_vectors,*,seconds=10,
                  initial_multipliers=None,penalty=.2):
    """Simultaneous vector updates for the inequality augmented Lagrangian."""
    began=time.perf_counter();deadline=began+seconds
    row=cut_matrix(source.cuts,len(edges));B=parity_matrix(parity_cuts,n)
    shift=np.asarray(row.sum(axis=1)).ravel()-2*np.array([c.rhs for c in source.cuts])
    original=np.array([float(weights[e]) for e in edges]);u,v=np.asarray(edges).T
    edge_lambda=np.array([float(Fraction(a,source.denominator)) for a in source.multipliers])
    parity_lambda=(np.zeros(len(parity_cuts)) if initial_multipliers is None
                   else np.asarray(initial_multipliers)/2**20)
    latest=initial_vectors.copy().ravel();rho=penalty
    iterations=0
    def evaluate(flat,gradient=True):
        V=flat.reshape(n,-1);norms=np.maximum(np.linalg.norm(V,axis=1)[:,None],1e-100);V=V/norms
        correlation=np.sum(V[u]*V[v],axis=1);BV=B@V
        ec=shift-row@correlation;pc=1-np.sum(BV*BV,axis=1)
        em=np.maximum(0,edge_lambda+rho*ec);pm=np.maximum(0,parity_lambda+rho*pc)
        if not gradient:return V,em,pm,ec,pc
        value=original@correlation+(em@em-edge_lambda@edge_lambda+pm@pm-parity_lambda@parity_lambda)/(2*rho)
        residual=original-row.T@em
        C=sp.csr_matrix((np.r_[residual,residual]/2,(np.r_[u,v],np.r_[v,u])),shape=(n,n))
        applied=C@V-B.T@(pm[:,None]*BV)
        return value,(2*(applied-V*np.sum(V*applied,axis=1)[:,None])/norms).ravel()
    def checkpoint(flat):
        nonlocal latest
        latest=flat.copy()
        if time.perf_counter()>=deadline-.8:raise StopIteration
    previous_violation=float('inf')
    while time.perf_counter()<deadline-.8 and iterations<60:
        result=minimize(evaluate,latest,jac=True,method='L-BFGS-B',callback=checkpoint,
                        options={'maxiter':80,'maxcor':5,'ftol':1e-9,'gtol':1e-6})
        latest=result.x
        V,edge_lambda,parity_lambda,ec,pc=evaluate(latest,False)
        latest=V.copy().ravel()
        iterations+=1
        if iterations%5==0:
            violation=max(np.max(ec,initial=0),np.max(pc,initial=0))
            if violation>previous_violation*.8:rho=min(2.,rho*1.5)
            previous_violation=violation
        if result.status==99:break
    source=make_certificate(source.cuts,-edge_lambda,edges,weights,Fraction())
    # Reuse the same exact witness construction, allowing a final dual refit.
    r=fit_parity(n,edges,weights,source,parity_cuts,latest.reshape(n,-1),
                 seconds=max(.3,deadline-time.perf_counter()),joint=False,
                 initial_multipliers=np.rint(parity_lambda*2**20).astype(np.int64).tolist())
    r.update(seconds=time.perf_counter()-began,augmented_iterations=iterations,
             violation=float(max(np.max(ec,initial=0),np.max(pc,initial=0))))
    return r
