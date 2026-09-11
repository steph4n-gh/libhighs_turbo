"""Research: exact local cut polytopes chosen in vector geometry."""
from fractions import Fraction
from itertools import combinations
import time
import numpy as np

from highs_turbo.ising_cuts import IsingCut,make_certificate,subgraph_cut,verify_cut


def regions(vectors,parity_cuts,*,size=8,limit=128):
    n=len(vectors)
    candidates=[]
    for cut in parity_cuts:
        nodes=[i for i,b in cut]
        if len(nodes)>size:continue
        vector=sum(b*vectors[i] for i,b in cut)
        candidates.append((float(vector@vector),nodes))
    seen=set();usage=np.zeros(n,dtype=int);chosen=[]
    for score,nodes in sorted(candidates):
        if any(usage[n]>=8 for n in nodes):continue
        strength=np.sum((vectors@vectors[nodes].T)**2,axis=1)
        for node in np.argsort(-strength):
            if int(node) not in nodes:nodes.append(int(node))
            if len(nodes)>=size:break
        key=tuple(sorted(nodes))
        if key not in seen:
            seen.add(key);chosen.append(key);usage[list(key)]+=1
        if len(chosen)>=limit:break
    return chosen


def extend_problem(edges,weights,source,parity_cuts,parity_multipliers,
                   vectors,*,size=8,limit=128,seconds=5,facets=False):
    began=time.perf_counter();deadline=began+seconds
    groups=regions(vectors,parity_cuts,size=size,limit=limit)
    expanded=sorted(set(edges)|{e for group in groups for e in combinations(group,2)}
                    |{tuple(sorted((u,v))) for cut in parity_cuts for (u,b),(v,c) in combinations(cut,2)})
    weights={e:weights.get(e,Fraction()) for e in expanded};location={e:i for i,e in enumerate(expanded)}
    cuts=[IsingCut(tuple(location[edges[i]] for i in cut.indices),cut.coefficients,cut.rhs,cut.kind)
          for cut in source.cuts]
    multipliers=[-Fraction(a,source.denominator) for a in source.multipliers]
    for cut,multiplier in zip(parity_cuts,parity_multipliers):
        terms=sorted((location[tuple(sorted((u,v)))],b*c) for (u,b),(v,c) in combinations(cut,2))
        cuts.append(IsingCut(tuple(i for i,c in terms),tuple(c for i,c in terms),
                             (sum(b for i,b in cut)**2-1)//4,'subgraph'))
        multipliers.append(-Fraction(2*multiplier,2**20))
    u,v=np.asarray(expanded).T;point=(1-np.sum(vectors[u]*vectors[v],axis=1))/2
    found=[]
    direction={e:Fraction() for e in expanded} if facets else weights
    for nodes in groups:
        if time.perf_counter()>=deadline:break
        indices=tuple(location[e] for e in combinations(nodes,2))
        cut=subgraph_cut(indices,expanded,direction,point,min(deadline,time.perf_counter()+.03),threads=1)
        if cut is not None:
            assert verify_cut(cut,expanded)
            found.append(cut)
    source=make_certificate(cuts+found,multipliers+[0]*len(found),expanded,weights,Fraction())
    # make_certificate omits zero multipliers; preserve candidates for fitting.
    from dataclasses import replace
    source=replace(source,cuts=source.cuts+tuple(found),multipliers=source.multipliers+(0,)*len(found))
    return expanded,weights,source,dict(seconds=time.perf_counter()-began,groups=len(groups),
                                        found=len(found),edges=len(expanded))
