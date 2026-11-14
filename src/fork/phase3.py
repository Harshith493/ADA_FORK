"""Label-free fork-risk features, tiny NumPy logistic model, and score metrics.

This module does not import or compute transition/regime labels. Its supervised
target is the miss indicator for the frozen model's approximate-90 interval.
"""
from __future__ import annotations

import numpy as np


def calendar_features(timestamp) -> np.ndarray:
    """Four cyclical features for time-of-day and day-of-week."""
    ts=np.datetime64(timestamp,'m')
    minute_of_day=int((ts.astype('datetime64[m]')-ts.astype('datetime64[D]'))/np.timedelta64(1,'m'))
    weekday=int((ts.astype('datetime64[D]').astype(int)+3)%7)  # Monday=0 for Unix epoch Thursday
    tod=2*np.pi*minute_of_day/1440
    dow=2*np.pi*weekday/7
    return np.asarray([np.sin(tod),np.cos(tod),np.sin(dow),np.cos(dow)],np.float32)


def kmeans_2(paths: np.ndarray, seed: int = 0, n_init: int = 5, max_iter: int = 50):
    """Small deterministic two-means; returns assignments, centers, gap, share, score."""
    x=np.asarray(paths,dtype=float)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('paths must be finite [samples, dimensions] with at least two samples')
    best=None
    for run in range(n_init):
        rng=np.random.default_rng(seed+run)
        first=int(rng.integers(0,len(x)))
        d2=np.sum((x-x[first])**2,axis=1)
        if np.max(d2)<1e-12:
            second=(first+1)%len(x)
        else:
            second=int(np.argmax(d2))
        centers=np.stack([x[first],x[second]])
        assignment=np.zeros(len(x),dtype=int)
        for _ in range(max_iter):
            new=np.argmin(((x[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
            if np.array_equal(new,assignment) and _>0: break
            assignment=new
            for c in range(2):
                if np.any(assignment==c): centers[c]=x[assignment==c].mean(axis=0)
                else: centers[c]=x[int(np.argmax(((x-centers[1-c])**2).sum(axis=1)))]
        residual=x-centers[assignment]
        loss=float(np.sum(residual**2))
        if best is None or loss<best[0]:
            best=(loss,assignment.copy(),centers.copy())
    loss,assignment,centers=best
    counts=np.bincount(assignment,minlength=2)
    if np.any(counts==0):
        return assignment,centers,0.0,0.0,0.0
    gap=float(np.linalg.norm(centers[0]-centers[1]))
    spread=float(np.sqrt(loss/max(len(x)*x.shape[1],1)))
    share=float(counts.min()/len(x))
    score=float(gap/(spread+1e-8))
    return assignment,centers,gap,share,score


def detector_features(forecast_q, neighbor_paths, context, sensor_mean, sensor_std, seed=0,
                      n_init=5, max_iter=50):
    """Four label-free features specified for the FORK detector."""
    q=np.asarray(forecast_q,dtype=float)
    paths=np.asarray(neighbor_paths,dtype=float)
    context=np.asarray(context,dtype=float)
    std=max(float(sensor_std),1e-6)
    normalized=(paths-float(sensor_mean))/std
    _,_,_,minority,score=kmeans_2(normalized,seed=seed,n_init=n_init,max_iter=max_iter)
    q10,q20,_,q80,q90=np.moveaxis(q,-1,0)
    q05=q10-.5*(q20-q10); q95=q90+.5*(q90-q80)
    spread=float((q95[11]-q05[11])/std)
    gap=float(abs(np.median(paths[:,11])-q[11,2])/std)
    return np.asarray([score,minority,spread,gap],dtype=float)


def baseline_scores(context, train_mean, train_std):
    """Label-free volatility, recent downward drop, and negative CUSUM scores."""
    x=np.asarray(context,dtype=float)
    std=max(float(train_std),1e-6)
    dif=np.diff(x)
    volatility=float(np.std(dif[-12:])/std)
    drop=float((np.mean(x[-15:-3])-np.mean(x[-3:]))/std)
    reference=float(np.mean(x[-36:-12]))
    deviations=(reference-x[-12:])-.5*std
    cusum=float(max(0.,np.max(np.cumsum(deviations)))/std)
    return np.asarray([volatility,drop,cusum],dtype=float)


def fit_logistic(features, misses, l2=0.1, max_iter=100):
    """Fit a tiny L2 logistic regression to model-miss outcomes only."""
    x=np.asarray(features,dtype=float); y=np.asarray(misses,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)<2 or not np.isin(y,[0,1]).all():
        raise ValueError('features and binary miss targets have incompatible shapes')
    mean=x.mean(axis=0); scale=x.std(axis=0); scale=np.where(scale<1e-8,1.,scale)
    z=(x-mean)/scale
    design=np.column_stack([np.ones(len(z)),z])
    w=np.zeros(design.shape[1]); reg=np.eye(len(w))*float(l2); reg[0,0]=0
    for _ in range(max_iter):
        logits=np.clip(design@w,-30,30)
        p=1/(1+np.exp(-logits))
        grad=design.T@(p-y)/len(y)+reg@w
        weight=np.maximum(p*(1-p),1e-6)
        hess=(design.T*weight)@design/len(y)+reg+np.eye(len(w))*1e-8
        step=np.linalg.solve(hess,grad)
        w-=step
        if np.max(np.abs(step))<1e-7: break
    return {'mean':mean,'scale':scale,'weights':w,'l2':float(l2)}


def predict_logistic(model, features):
    x=np.asarray(features,dtype=float)
    z=(x-model['mean'])/model['scale']
    design=np.column_stack([np.ones(len(z)),z])
    logits=np.clip(design@model['weights'],-30,30)
    return 1/(1+np.exp(-logits))


def binary_log_loss(target, probability):
    y=np.asarray(target,dtype=float); p=np.clip(np.asarray(probability,dtype=float),1e-8,1-1e-8)
    if y.shape!=p.shape or not np.isin(y,[0,1]).all(): raise ValueError('invalid binary log-loss inputs')
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def threshold_by_f1(scores, target):
    s=np.asarray(scores,dtype=float); y=np.asarray(target,dtype=int)
    candidates=np.unique(s)
    best=(-1.,float(np.median(s)))
    for threshold in candidates:
        pred=s>=threshold
        tp=np.sum(pred&(y==1)); fp=np.sum(pred&(y==0)); fn=np.sum((~pred)&(y==1))
        f1=2*tp/max(2*tp+fp+fn,1)
        if f1>best[0]: best=(float(f1),float(threshold))
    return best[1]


def precision_recall_curve(target, score):
    y=np.asarray(target,dtype=int); s=np.asarray(score,dtype=float)
    if y.shape!=s.shape or len(y)==0 or not np.isin(y,[0,1]).all():
        raise ValueError('invalid precision-recall inputs')
    order=np.argsort(-s,kind='stable'); ys=y[order]; ss=s[order]
    tp=np.cumsum(ys); fp=np.cumsum(1-ys); total=max(int(y.sum()),1)
    # Collapse ties before measuring area; row order must not alter AUPRC.
    group_end=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    group_tp=tp[group_end]; group_fp=fp[group_end]
    recall=np.r_[0.,group_tp/total]
    precision=np.r_[1.,group_tp/np.maximum(group_tp+group_fp,1)]
    thresholds=np.r_[np.inf,ss[group_end]]
    # Average precision (stepwise area under the empirical PR curve).
    auprc=float(np.sum((recall[1:]-recall[:-1])*precision[1:]))
    return precision,recall,thresholds,auprc


def false_alarm_rate(target_transition, flags):
    y=np.asarray(target_transition,dtype=bool); f=np.asarray(flags,dtype=bool)
    calm=~y
    return float(np.mean(f[calm])) if np.any(calm) else float('nan')


def generate_synthetic(n=600, context_steps=168, horizon_steps=12, seed=90210,
                       jump_mph=24., coupling=.8):
    """Independent two-regime episodes with observable odds and coupled branches.

    Fork outcomes are returned separately so callers can reserve them for scoring.
    Each episode has two sensors whose fork indicators share a latent event with
    the requested coupling; each sensor has marginal fork odds equal to p.
    """
    rng=np.random.default_rng(seed)
    odds=np.asarray([.05,.2,.4,.65,.85])
    p=rng.choice(odds,size=n)
    contexts=np.empty((n,2,context_steps),np.float32)
    futures=np.empty((n,2,horizon_steps),np.float32)
    outcomes=np.zeros((n,2),int)
    base=np.empty((n,2),float)
    for i in range(n):
        shared=bool(rng.random()<p[i])
        regime=int(rng.integers(0,2))
        for s in range(2):
            level=(26. if regime==0 else 58.) + (s*1.5)
            noise=.5+4.5*p[i]
            drift=rng.normal(0,.015)
            eps=rng.normal(0,noise,context_steps)
            contexts[i,s]=level+drift*np.arange(-context_steps,0)+eps
            base[i,s]=level
            fork=shared if rng.random()<coupling else bool(rng.random()<p[i])
            outcomes[i,s]=int(fork)
            destination=level+(jump_mph if level<40 else -jump_mph)
            ramp=np.linspace(0,1,horizon_steps)
            if fork:
                futures[i,s]=level+(destination-level)*ramp+rng.normal(0,noise*.2,horizon_steps)
            else:
                futures[i,s]=level+drift*np.arange(1,horizon_steps+1)+rng.normal(0,noise*.25,horizon_steps)
    return {'contexts':contexts,'futures':futures,'fork_odds':p,'fork_outcomes':outcomes,
            'base_regime':base,'jump_mph':float(jump_mph),'coupling':float(coupling)}
