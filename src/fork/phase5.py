"""Prequential conformal scores and group-specific ACI state."""
from __future__ import annotations
import heapq
import math
import numpy as np

def _quantile_sorted(x, q):
    if len(x)==1: return float(x[0])
    pos=float(np.clip(q,0.,1.))*(len(x)-1); i=int(pos); f=pos-i
    if i+1>=len(x): return float(x[-1])
    return float(x[i]*(1-f)+x[i+1]*f)

def branch_bounds(branch_paths, branch_counts, weights, tau, min_weight=.1, sorted_samples=False):
    paths=np.asarray(branch_paths,dtype=float); counts=np.asarray(branch_counts,dtype=int); w=np.asarray(weights,dtype=float)
    if paths.ndim!=3 or paths.shape[0]!=len(counts) or len(w)!=len(counts) or not 0<=tau<=1:
        raise ValueError('invalid branch samples, weights, or level')
    active=[j for j,n in enumerate(counts) if n>0 and w[j]>=min_weight]
    if not active: active=[int(np.argmax(w))]
    low=np.empty(paths.shape[2]); high=np.empty_like(low)
    for h in range(paths.shape[2]):
        low[h],high[h]=_bounds_at(paths,counts,active,h,tau,sorted_samples)
    return low,high

def _bounds_at(paths,counts,active,h,tau,sorted_samples=False):
    lows=[]; highs=[]
    for j in active:
        v=paths[j,:counts[j],h] if sorted_samples else np.sort(paths[j,:counts[j],h])
        lows.append(_quantile_sorted(v,(1-tau)/2)); highs.append(_quantile_sorted(v,(1+tau)/2))
    return min(lows),max(highs)

def central_bounds_at(branch_paths, branch_counts, weights, h, tau, min_weight=.1, sorted_samples=False):
    paths=np.asarray(branch_paths,dtype=float); counts=np.asarray(branch_counts,dtype=int); w=np.asarray(weights,dtype=float)
    active=[j for j,n in enumerate(counts) if n>0 and w[j]>=min_weight]
    if not active: active=[int(np.argmax(w))]
    return _bounds_at(paths,counts,active,int(h),float(tau),sorted_samples)

def branch_nonconformity(branch_paths, branch_counts, weights, actual, min_weight=.1, steps=20, sorted_samples=False):
    """Minimum central level covering each step; out-of-support scores cap at one."""
    y=np.asarray(actual,dtype=float)
    if y.ndim!=1 or y.shape[0]!=np.asarray(branch_paths).shape[2]: raise ValueError('actual path shape mismatch')
    scores=np.empty(len(y),float)
    p=np.asarray(branch_paths,dtype=float); counts=np.asarray(branch_counts,dtype=int); w=np.asarray(weights,dtype=float)
    active=[j for j,n in enumerate(counts) if n>0 and w[j]>=min_weight]
    if not active: active=[int(np.argmax(w))]
    for h,value in enumerate(y):
        full_lo,full_hi=_bounds_at(p,counts,active,h,1.,sorted_samples)
        if value<full_lo or value>full_hi: scores[h]=1.; continue
        lo=0.; hi=1.
        for _ in range(steps):
            mid=(lo+hi)/2; lower,upper=_bounds_at(p,counts,active,h,mid,sorted_samples)
            if lower<=value<=upper: hi=mid
            else: lo=mid
        scores[h]=hi
    return scores

def finite_sample_level(buffer, alpha_state, target_alpha=.1):
    values=np.asarray(buffer,dtype=float)
    if len(values)==0: return 1.-target_alpha
    rank=int(math.ceil((len(values)+1)*(1.-alpha_state)))
    rank=min(max(rank,1),len(values))
    return float(np.sort(values)[rank-1])

class ConformalState:
    """Independent ACI alpha and score buffers for every gate group/stream."""
    def __init__(self, streams=4, initial_alpha=.1):
        self.streams=int(streams); self.initial_alpha=float(initial_alpha); self.states={}
    def _group(self, group):
        if group not in self.states:
            self.states[group]={'alpha':np.full(self.streams,self.initial_alpha,float),
                                'buffers':[[] for _ in range(self.streams)]}
        return self.states[group]
    def level(self, group, stream, target_alpha=.1):
        state=self._group(group); a=float(state['alpha'][stream])
        return finite_sample_level(state['buffers'][stream],a,target_alpha)
    def update(self, group, stream, score, miss, gamma, target_alpha=.1, clip=(.01,.99)):
        state=self._group(group); state['buffers'][stream].append(float(score))
        new=state['alpha'][stream]+gamma*(target_alpha-int(bool(miss)))
        state['alpha'][stream]=np.clip(new,*clip)
        return float(state['alpha'][stream])

class MaturityQueue:
    """Apply score/error updates only when outcome maturity is reached."""
    def __init__(self): self._heap=[]; self._order=0
    def push(self, maturity, group, stream, score=None, miss=None, reference=None, bounds=None):
        self._order+=1; heapq.heappush(self._heap,(int(maturity),self._order,group,int(stream),score,miss,reference,bounds))
    def mature(self, now, state, gamma, target_alpha=.1, clip=(.01,.99), observe=None):
        done=[]
        while self._heap and self._heap[0][0]<=int(now):
            maturity,_,group,stream,score,miss,reference,bounds=heapq.heappop(self._heap)
            if reference is not None:
                if observe is None: raise ValueError('matured outcomes require an observer')
                score,miss=observe(reference,stream,bounds)
            state.update(group,stream,score,miss,gamma,target_alpha,clip); done.append(maturity)
        return done

