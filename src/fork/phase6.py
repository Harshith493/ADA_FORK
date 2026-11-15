"""Small deterministic helpers for Phase 6 calibration comparisons."""
from __future__ import annotations
import numpy as np

def day_block_bootstrap_ci(day_ids, metric_specs, day_count, resamples=1000, seed=6060):
    """Percentile intervals from resampling whole day blocks with replacement.

    Each metric spec is (row_mask, row_values); its statistic is the mean over
    included rows, with each sampled day replicated by its bootstrap count.
    """
    ids=np.asarray(day_ids,dtype=int)
    if ids.ndim!=1 or np.any(ids<0) or np.any(ids>=day_count) or resamples<1:
        raise ValueError('invalid day block inputs')
    prepared={}
    for name,(mask,values) in metric_specs.items():
        m=np.asarray(mask,dtype=bool); v=np.asarray(values,dtype=float)
        if m.shape!=ids.shape or v.shape!=ids.shape: raise ValueError('metric arrays must match day ids')
        prepared[name]=(m,v)
    rng=np.random.default_rng(seed); draws={name:np.empty(resamples,float) for name in prepared}
    for b in range(resamples):
        counts=np.bincount(rng.integers(0,day_count,size=day_count),minlength=day_count)
        row_weights=counts[ids]
        for name,(mask,values) in prepared.items():
            w=row_weights*mask; denom=w.sum()
            draws[name][b]=float(np.sum(w*values)/denom) if denom else np.nan
    return {name:(float(np.nanpercentile(vals,2.5)),float(np.nanpercentile(vals,97.5))) for name,vals in draws.items()}

def interpolate_at(x, y, target):
    """Linear interpolation only when target lies inside observed x range."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    valid=np.isfinite(x)&np.isfinite(y); x=x[valid]; y=y[valid]
    if len(x)<2: return None
    order=np.argsort(x); x=x[order]; y=y[order]
    unique=np.unique(x)
    if len(unique)<2 or target<unique[0] or target>unique[-1]: return None
    uy=np.asarray([np.mean(y[x==v]) for v in unique])
    return float(np.interp(target,unique,uy))

def minority_reliability(weights, counts, paths, actual, bins=5):
    """Reliability of minority mass for nearest-centre outcome; labels are absent."""
    w=np.asarray(weights); c=np.asarray(counts); p=np.asarray(paths); y=np.asarray(actual)
    pred=[]; hit=[]
    for i in range(len(y)):
        active=np.flatnonzero(c[i]>0)
        if len(active)<2: continue
        minority=active[np.argmin(w[i,active])]; majority=active[np.argmax(w[i,active])]
        centres=np.asarray([np.mean(p[i,j,:c[i,j]],axis=0) for j in (minority,majority)])
        pred.append(float(w[i,minority])); hit.append(float(np.linalg.norm(y[i]-centres[0]) < np.linalg.norm(y[i]-centres[1])))
    pred=np.asarray(pred); hit=np.asarray(hit); edges=np.linspace(0,.5,bins+1); rows=[]
    for j in range(bins):
        m=(pred>=edges[j]) & ((pred<edges[j+1]) if j<bins-1 else (pred<=edges[j+1]))
        rows.append((float(pred[m].mean()) if np.any(m) else float('nan'),float(hit[m].mean()) if np.any(m) else float('nan'),int(m.sum())))
    return rows, float(np.mean((pred-hit)**2)) if len(hit) else float('nan'),pred,hit
