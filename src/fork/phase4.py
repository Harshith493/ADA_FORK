"""Label-free weighted branches and Gaussian-copula path sampling."""
from __future__ import annotations

import math
import numpy as np


def smoothed_weights(counts, alpha):
    counts=np.asarray(counts,dtype=float)
    if counts.ndim!=1 or len(counts)==0 or np.any(counts<0) or alpha<0:
        raise ValueError('invalid branch counts or smoothing')
    weights=(counts+alpha)/(counts.sum()+alpha*len(counts))
    return weights/weights.sum()


def corr_from_residuals(residuals, dimensions=12):
    x=np.asarray(residuals,dtype=float)
    if x.ndim!=2 or x.shape[1]!=dimensions:
        raise ValueError('residuals must have shape [samples, dimensions]')
    if len(x)<2:
        return np.eye(dimensions)
    centered=x-x.mean(axis=0,keepdims=True)
    scale=np.sqrt(np.mean(centered*centered,axis=0))
    safe=np.where(scale>1e-10,scale,1.)
    standardized=centered/safe
    corr=(standardized.T@standardized)/max(len(x)-1,1)
    corr=np.nan_to_num(corr,nan=0.,posinf=0.,neginf=0.)
    corr=(corr+corr.T)*.5
    np.fill_diagonal(corr,1.)
    # Numerical projection keeps the matrix usable by Cholesky while retaining unit diagonal.
    eig,vec=np.linalg.eigh(corr)
    corr=(vec*np.maximum(eig,1e-7))@vec.T
    scale=np.sqrt(np.maximum(np.diag(corr),1e-12))
    corr=corr/scale[:,None]/scale[None,:]
    np.fill_diagonal(corr,1.)
    return corr


def shrink_correlation(local_corr, pooled_corr, shrinkage):
    local=np.asarray(local_corr,dtype=float); pooled=np.asarray(pooled_corr,dtype=float)
    if local.shape!=pooled.shape or local.ndim!=2 or local.shape[0]!=local.shape[1]:
        raise ValueError('correlation matrices must be square and have matching shapes')
    if not 0<=shrinkage<=1:
        raise ValueError('shrinkage must lie in [0,1]')
    result=(1-shrinkage)*local+shrinkage*pooled
    result=(result+result.T)*.5
    np.fill_diagonal(result,1.)
    return result


def _normal_cdf(values):
    """Vectorized Abramowitz-Stegun normal CDF approximation."""
    x=np.asarray(values,dtype=float); a=np.abs(x)
    t=1/(1+.2316419*a)
    poly=t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
    tail=np.exp(-.5*a*a)/np.sqrt(2*np.pi)*poly
    return np.clip(np.where(x>=0,1-tail,tail),1e-8,1-1e-8)


def gaussian_copula_uniforms(correlation, count, seed):
    corr=np.asarray(correlation,dtype=float)
    if corr.ndim!=2 or corr.shape[0]!=corr.shape[1]:
        raise ValueError('copula correlation must be a square matrix')
    if count<1 or not np.isfinite(corr).all():
        raise ValueError('invalid copula inputs')
    if not np.allclose(corr,corr.T,atol=1e-7) or not np.allclose(np.diag(corr),1.,atol=1e-6):
        raise ValueError('correlation must be symmetric with unit diagonal')
    chol=np.linalg.cholesky(corr+np.eye(len(corr))*1e-8)
    z=np.random.default_rng(seed).normal(size=(count,len(corr)))@chol.T
    return _normal_cdf(z)


def quantile_copula_paths(quantiles, levels, correlation, count=200, seed=0):
    q=np.asarray(quantiles,dtype=float); levels=np.asarray(levels,dtype=float)
    if q.ndim!=2 or q.shape[1]!=len(levels) or q.shape[0]!=np.asarray(correlation).shape[0]:
        raise ValueError('quantiles and correlation dimensions do not match')
    uniforms=gaussian_copula_uniforms(correlation,count,seed)
    return np.stack([np.interp(uniforms[:,h],levels,q[h],left=q[h,0],right=q[h,-1])
                     for h in range(q.shape[0])],axis=1)


def residual_copula_paths(center, residuals, correlation, count=200, seed=0):
    center=np.asarray(center,dtype=float); residuals=np.asarray(residuals,dtype=float)
    corr=np.asarray(correlation,dtype=float)
    if center.ndim!=1 or corr.shape!=(len(center),len(center)):
        raise ValueError('center and correlation dimensions do not match')
    if residuals.ndim!=2 or residuals.shape[1]!=len(center) or len(residuals)==0:
        raise ValueError('residual bank has incompatible shape')
    uniforms=gaussian_copula_uniforms(corr,count,seed)
    if len(residuals)==1:
        return np.repeat(center[None,:]+residuals,count,axis=0)
    ranks=(np.arange(len(residuals))+.5)/len(residuals)
    return center[None,:]+np.stack([
        np.interp(uniforms[:,h],ranks,np.sort(residuals[:,h]))
        for h in range(len(center))
    ],axis=1)


def cluster_branches(neighbor_paths, chronos_median, sensor_mean, sensor_std, seed=0):
    """Cluster neighbors and return centers/residual banks; no labels are accepted."""
    paths=np.asarray(neighbor_paths,dtype=float)
    median=np.asarray(chronos_median,dtype=float)
    std=max(float(sensor_std),1e-6); mean=float(sensor_mean)
    if paths.ndim!=2 or paths.shape[1]!=12 or median.shape!=(12,):
        raise ValueError('expected 12-step neighbor paths and Chronos median')
    z=(paths-mean)/std
    assignment,_,_,_,_=__import__('fork.phase3',fromlist=['kmeans_2']).kmeans_2(z,seed=seed,n_init=3,max_iter=30)
    centers=[]; residuals=[]; counts=[]
    for branch in range(2):
        mask=assignment==branch
        if np.any(mask):
            center=paths[mask].mean(axis=0)
            centers.append(center); residuals.append(paths[mask]-center); counts.append(int(mask.sum()))
    anchor=int(np.argmin([np.linalg.norm((c-median)/std) for c in centers]))
    return {'centers':np.asarray(centers),'residuals':residuals,'counts':np.asarray(counts),
            'anchor':anchor,'assignment':assignment}


def active_branches(branch, alpha, collapse_threshold):
    weights=smoothed_weights(branch['counts'],alpha)
    if len(weights)==2 and float(weights.min())<collapse_threshold:
        keep=int(np.argmax(weights))
        active=[keep]; weights=np.zeros(2,float); weights[keep]=1.
    else:
        active=list(range(len(weights)))
    return active,weights


def allocate_path_counts(weights, total=200):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or np.any(w<0) or not np.isclose(w.sum(),1.) or total<1:
        raise ValueError('weights must be nonnegative and sum to one')
    raw=w*total; counts=np.floor(raw).astype(int)
    for j in np.argsort(-(raw-counts))[:total-int(counts.sum())]: counts[j]+=1
    return counts


def sample_branch_paths(branch, q, pooled_corr, alpha, collapse_threshold, shrinkage,
                        count=200, seed=0, chronos_anchor=True):
    active,weights=active_branches(branch,alpha,collapse_threshold)
    counts=allocate_path_counts(weights,count)
    q=np.asarray(q,dtype=float); local_samples={}; parts=[]
    for j in active:
        residuals=branch['residuals'][j]
        local=corr_from_residuals(residuals,12)
        corr=shrink_correlation(local,pooled_corr,shrinkage)
        use_chronos=chronos_anchor and j==branch['anchor']
        if use_chronos:
            paths=quantile_copula_paths(q,[.1,.2,.5,.8,.9],corr,int(counts[j]),seed+31*j)
        else:
            paths=residual_copula_paths(branch['centers'][j],residuals,corr,
                                         int(counts[j]),seed+31*j)
        local_samples[j]=paths
        parts.append(paths)
    combined=np.concatenate(parts,axis=0) if parts else np.empty((0,12))
    return combined,local_samples,weights,counts


def split_interval(branch_samples, weights, minimum_weight=.05, alpha=.1):
    active=[j for j,w in enumerate(weights) if w>=minimum_weight and j in branch_samples]
    if not active:
        active=[max(branch_samples,key=lambda j:weights[j])]
    lower=[]; upper=[]
    for j in active:
        lo,hi=np.quantile(branch_samples[j],[alpha/2,1-alpha/2],axis=0)
        lower.append(lo); upper.append(hi)
    return np.min(lower,axis=0),np.max(upper,axis=0)


def energy_score(paths, actual, seed=0, pair_count=2000):
    x=np.asarray(paths,dtype=float); y=np.asarray(actual,dtype=float)
    if x.ndim!=2 or y.shape!=(x.shape[1],) or len(x)<2:
        raise ValueError('energy score requires [samples,horizon] and one matching path')
    first=float(np.mean(np.linalg.norm(x-y[None,:],axis=1)))
    rng=np.random.default_rng(seed); a=rng.integers(len(x),size=pair_count); b=rng.integers(len(x),size=pair_count)
    pair=float(np.mean(np.linalg.norm(x[a]-x[b],axis=1)))
    return first-.5*pair


def variogram_score(paths, actual, p=.5):
    x=np.asarray(paths,dtype=float); y=np.asarray(actual,dtype=float)
    if x.ndim!=2 or y.shape!=(x.shape[1],) or not 0<p<=2:
        raise ValueError('invalid variogram score inputs')
    h=x.shape[1]; i,j=np.triu_indices(h,1)
    observed=np.abs(y[i]-y[j])**p
    forecast=np.mean(np.abs(x[:,i]-x[:,j])**p,axis=0)
    return float(np.mean((observed-forecast)**2))
