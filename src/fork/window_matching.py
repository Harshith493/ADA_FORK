"""Label-free nearest historical windows, fitted on training data only."""
from __future__ import annotations

import numpy as np


class WindowMatcher:
    """Exact Euclidean matching after train-only per-sensor standardization."""

    def __init__(self, train_values: np.ndarray, context_steps: int, horizon_steps: int, valid_origins=None):
        values = np.asarray(train_values, dtype=np.float32)
        if values.ndim != 1 or len(values) <= context_steps + horizon_steps:
            raise ValueError("training series is too short")
        self.mean = float(values.mean())
        self.std = max(float(values.std()), 1e-6)
        normalized = (values - self.mean) / self.std
        origins = (np.arange(context_steps, len(values) - horizon_steps + 1)
                   if valid_origins is None else np.asarray(valid_origins, dtype=int))
        origins = origins[(origins >= context_steps) & (origins + horizon_steps <= len(values))]
        self.windows = np.stack([normalized[o-context_steps:o] for o in origins])
        self.futures = np.stack([values[o:o+horizon_steps] for o in origins])
        self.norms = np.einsum("ij,ij->i", self.windows, self.windows)

    def query(self, context: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest future paths and their training-window indices."""
        if not 1 <= k <= len(self.windows):
            raise ValueError("k must be within the number of training windows")
        q = (np.asarray(context, dtype=np.float32) - self.mean) / self.std
        if q.shape != (self.windows.shape[1],):
            raise ValueError("query context length differs from fitted context")
        distances = self.norms - 2 * (self.windows @ q) + float(q @ q)
        ids = np.argpartition(distances, k-1)[:k]
        ids = ids[np.argsort(distances[ids], kind="stable")]
        return self.futures[ids], ids


def empirical_interval(paths: np.ndarray, lower_q: float = .05, upper_q: float = .95) -> tuple[np.ndarray, np.ndarray]:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] == 0:
        raise ValueError("paths must have shape [samples, horizon]")
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("invalid quantiles")
    return np.quantile(paths, lower_q, axis=0), np.quantile(paths, upper_q, axis=0)
