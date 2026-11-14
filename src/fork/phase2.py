"""Phase 2 calibration and score utilities. Labels stay in reporting code."""
from __future__ import annotations
import numpy as np


def forecast_contexts(forecaster, contexts):
    """Phase 2 prediction boundary: future values and labels are not accepted."""
    x=np.asarray(contexts,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError("forecaster accepts finite context arrays only")
    return np.asarray(forecaster.forecast(x),dtype=np.float32)


def interval_score(actual, lower, upper, alpha=.1):
    y, lo, hi = np.asarray(actual, float), np.asarray(lower, float), np.asarray(upper, float)
    if y.shape != lo.shape or y.shape != hi.shape or not 0 < alpha < 1 or np.any(lo > hi):
        raise ValueError("invalid interval score inputs")
    return (hi-lo) + (2/alpha)*(lo-y)*(y < lo) + (2/alpha)*(y-hi)*(y > hi)


def aci_update(alpha, miss, gamma, target_alpha):
    if miss not in (0, 1) or gamma < 0:
        raise ValueError("miss must be binary and gamma nonnegative")
    return float(np.clip(alpha + gamma*(target_alpha-miss), 0.001, .999))


def widen_bounds(lower, upper, alpha, target_alpha):
    """Scale symmetric interval radius to the current ACI miscoverage level."""
    center = (np.asarray(lower) + np.asarray(upper))/2
    radius = (np.asarray(upper) - np.asarray(lower))/2
    # A smaller online miscoverage state requests more coverage, so bounds widen.
    scale = np.sqrt(target_alpha/max(alpha, 1e-4))
    return center-radius*scale, center+radius*scale


def mix_samples(model_samples, history_samples, weight, rng):
    """Replace a fixed fraction of model paths with empirical history paths."""
    model = np.asarray(model_samples)
    hist = np.asarray(history_samples)
    if model.ndim != 2 or hist.ndim != 2 or model.shape[1] != hist.shape[1] or not 0 <= weight <= 1:
        raise ValueError("incompatible samples or weight")
    n_hist = int(round(len(model)*weight))
    n_model = len(model)-n_hist
    m = model[rng.choice(len(model), n_model, replace=n_model > len(model))] if n_model else model[:0]
    h = hist[rng.choice(len(hist), n_hist, replace=n_hist > len(hist))] if n_hist else hist[:0]
    return np.concatenate([m, h], axis=0)


def quantiles_to_samples(qvalues, levels, count=100):
    """Deterministic inverse-CDF interpolation; outer tails held constant."""
    q = np.asarray(qvalues, dtype=float)
    levels = np.asarray(levels, dtype=float)
    u = (np.arange(count)+.5)/count
    if q.ndim != 2 or q.shape[1] != len(levels):
        raise ValueError("qvalues must have shape [horizon, quantile]")
    return np.stack([np.interp(u, levels, q[h], left=q[h,0], right=q[h,-1]) for h in range(q.shape[0])], axis=1)
