"""Phase 1: reproduce Chronos-Bolt transition coverage on PEMS-BAY."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGIMES = ("free-flow", "transition", "congested")
METHOD_APPROX_90 = "approximate nominal 90%"
METHOD_NATIVE_80 = "native 80% (P10-P90)"
QUANTILES = np.array([0.1, 0.2, 0.5, 0.8, 0.9], dtype=np.float32)


class Forecaster(Protocol):
    """The prediction boundary accepts context values only, never labels."""

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        """Return [batch, horizon, 5] quantiles for the supplied contexts."""


def coverage_rate(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical inclusive interval coverage for equally shaped arrays."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if actual.shape != lower.shape or actual.shape != upper.shape:
        raise ValueError("actual, lower, and upper must have the same shape")
    if actual.size == 0:
        raise ValueError("coverage is undefined for an empty sample")
    if not (np.isfinite(actual).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("coverage inputs must be finite")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def label_target_period(
    future_speeds: np.ndarray,
    free_flow_mph: float = 55.0,
    congested_mph: float = 25.0,
) -> str:
    """Classify an observed target period for scoring only."""
    speeds = np.asarray(future_speeds, dtype=float)
    if speeds.ndim != 1 or speeds.size == 0 or not np.isfinite(speeds).all():
        raise ValueError("target period must be a non-empty finite 1D array")
    if np.all(speeds > free_flow_mph):
        return "free-flow"
    if np.all(speeds < congested_mph):
        return "congested"
    return "transition"


def approximate_90_bounds(native_quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly extend q10-q20 and q80-q90 slopes to q05 and q95."""
    values = np.asarray(native_quantiles, dtype=float)
    if values.shape[-1] != len(QUANTILES):
        raise ValueError("expected final quantile dimension [q10, q20, q50, q80, q90]")
    q10, q20, _, q80, q90 = np.moveaxis(values, -1, 0)
    lower = q10 - 0.5 * (q20 - q10)
    upper = q90 + 0.5 * (q90 - q80)
    return lower, upper


def coverage_mask(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boolean coverage mask, exposed separately for metric tests."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if actual.shape != lower.shape or actual.shape != upper.shape:
        raise ValueError("actual, lower, and upper must have the same shape")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    return (actual >= lower) & (actual <= upper)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_period(data_path: Path, start: str, end_exclusive: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(data_path, parse_dates=[0])
    frame = frame.rename(columns={frame.columns[0]: "timestamp"})
    frame = frame.loc[
        (frame["timestamp"] >= pd.Timestamp(start)) & (frame["timestamp"] < pd.Timestamp(end_exclusive))
    ].reset_index(drop=True)
    sensor_columns = [column for column in frame.columns if column != "timestamp"]
    if frame.empty or len(sensor_columns) != 325:
        raise ValueError(f"Expected Jan-May PEMS-BAY with 325 sensors; found {len(sensor_columns)} sensors")
    if frame[sensor_columns].isna().any().any():
        raise ValueError("PEMS-BAY contains missing speed values; explicit handling is required")
    return frame, sensor_columns


def _split_indices(row_count: int, train_fraction: float, calibration_fraction: float) -> tuple[int, int]:
    train_end = int(np.floor(row_count * train_fraction))
    calibration_end = int(np.floor(row_count * (train_fraction + calibration_fraction)))
    if not (0 < train_end < calibration_end < row_count):
        raise ValueError("invalid chronological split fractions")
    return train_end, calibration_end


def _sensor_groups(
    train_values: np.ndarray,
    sensor_columns: list[str],
    top_count: int,
    sample_count: int,
    seeds: list[int],
    congested_mph: float,
) -> tuple[list[list[str]], list[str]]:
    rates = np.mean(train_values < congested_mph, axis=0)
    # Numeric tie-break makes rankings independent of CSV column order.
    ranked_indices = sorted(range(len(sensor_columns)), key=lambda i: (-rates[i], int(sensor_columns[i])))
    top = [sensor_columns[i] for i in ranked_indices[:top_count]]
    groups = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        selected = rng.choice(top, size=sample_count, replace=False).tolist()
        groups.append(selected)
    return groups, top


def _valid_test_origins(
    timestamps: pd.Series,
    test_start: int,
    test_end: int,
    context_steps: int,
    max_horizon: int,
    interval_minutes: int,
) -> np.ndarray:
    expected = pd.Timedelta(minutes=interval_minutes)
    times = pd.DatetimeIndex(timestamps)
    origins = []
    for origin in range(max(test_start, context_steps), test_end - max_horizon + 1):
        first = origin - context_steps
        last = origin + max_horizon - 1
        if np.all(np.diff(times[first : last + 1].asi8) == expected.value):
            origins.append(origin)
    return np.asarray(origins, dtype=int)


def select_window_origins(valid_origins: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(valid_origins) < count:
        raise ValueError(f"need {count} valid test windows; found {len(valid_origins)}")
    selected = np.random.default_rng(seed).choice(valid_origins, size=count, replace=False)
    return np.sort(selected)


class ChronosBoltCPU:
    """Small adapter whose only forecasting input is the past context array."""

    def __init__(self, model_id: str, torch_threads: int):
        import torch
        from chronos import BaseChronosPipeline

        torch.set_num_threads(torch_threads)
        self._torch = torch
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", local_files_only=True
        )

    def forecast(self, contexts: np.ndarray) -> np.ndarray:
        contexts = np.asarray(contexts, dtype=np.float32)
        if contexts.ndim != 2 or not np.isfinite(contexts).all():
            raise ValueError("forecaster accepts only a finite [batch, context] array")
        predicted, _ = self._pipeline.predict_quantiles(
            inputs=self._torch.as_tensor(contexts),
            prediction_length=12,
            quantile_levels=QUANTILES.tolist(),
        )
        return predicted.detach().cpu().numpy()


def _cache_metadata(config: dict, data_sha256: str) -> dict:
    return {
        "format": "fork-phase1-cache-v1",
        "data_sha256": data_sha256,
        "model_id": config["model_id"],
        "config": config,
    }


def _generate_forecasts(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    groups: list[list[str]],
    origins: np.ndarray,
    context_steps: int,
    max_horizon: int,
    forecaster: Forecaster,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[sensor_columns].to_numpy(dtype=np.float32)
    forecasts = np.empty((len(groups), len(groups[0]), len(origins), max_horizon, len(QUANTILES)), dtype=np.float32)
    actuals = np.empty((len(groups), len(groups[0]), len(origins), max_horizon), dtype=np.float32)
    sensor_index = {sensor: i for i, sensor in enumerate(sensor_columns)}
    for group_index, sensors in enumerate(groups):
        indices = [sensor_index[sensor] for sensor in sensors]
        for window_index, origin in enumerate(origins):
            # Only observations strictly before `origin` cross the forecasting boundary.
            contexts = values[origin - context_steps : origin, indices].T.copy()
            batch_forecasts = np.asarray(forecaster.forecast(contexts), dtype=np.float32)
            expected_shape = (len(sensors), max_horizon, len(QUANTILES))
            if batch_forecasts.shape != expected_shape:
                raise ValueError(f"forecaster returned {batch_forecasts.shape}, expected {expected_shape}")
            forecasts[group_index, :, window_index] = batch_forecasts
            # Future targets and their labels are kept outside the forecaster call.
            actuals[group_index, :, window_index] = values[origin : origin + max_horizon, indices].T
    return forecasts, actuals


def _metric_rows(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    horizons: list[int],
    seeds: list[int],
    config: dict,
) -> list[dict]:
    rows: list[dict] = []
    labels_by_horizon = {}
    for horizon in horizons:
        labels = np.empty(actuals.shape[:3], dtype="U12")
        for group in range(actuals.shape[0]):
            for sensor in range(actuals.shape[1]):
                for window in range(actuals.shape[2]):
                    labels[group, sensor, window] = label_target_period(
                        actuals[group, sensor, window, :horizon],
                        free_flow_mph=config["free_flow_mph"],
                        congested_mph=config["congested_mph"],
                    )
        labels_by_horizon[horizon] = labels

    for horizon in horizons:
        step = horizon - 1
        native_lower = forecasts[..., step, 0]
        native_upper = forecasts[..., step, 4]
        approximate_lower, approximate_upper = approximate_90_bounds(forecasts[..., step, :])
        actual = actuals[..., step]
        for method, lower, upper, nominal in (
            (METHOD_APPROX_90, approximate_lower, approximate_upper, 0.9),
            (METHOD_NATIVE_80, native_lower, native_upper, 0.8),
        ):
            for group_name, group_ids in [(str(seed), [i]) for i, seed in enumerate(seeds)] + [("all", list(range(len(seeds))))]:
                for regime in REGIMES:
                    mask = labels_by_horizon[horizon][group_ids] == regime
                    group_actual = actual[group_ids][mask]
                    group_lower = lower[group_ids][mask]
                    group_upper = upper[group_ids][mask]
                    rows.append(
                        {
                            "seed": group_name,
                            "horizon_steps": horizon,
                            "horizon_minutes": horizon * config["sampling_interval_minutes"],
                            "regime": regime,
                            "interval": method,
                            "nominal_coverage": nominal,
                            "n": int(group_actual.size),
                            "coverage": coverage_rate(group_actual, group_lower, group_upper) if group_actual.size else np.nan,
                            "mean_width_mph": float(np.mean(group_upper - group_lower)) if group_actual.size else np.nan,
                        }
                    )
    return rows


def _plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    aggregate = metrics.loc[metrics["seed"] == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    regime_positions = np.arange(len(REGIMES))
    bar_width = 0.34
    for axis, horizon in zip(axes, (3, 6, 12)):
        subset = aggregate.loc[aggregate["horizon_steps"] == horizon]
        for offset, method, color, label in (
            (-bar_width / 2, METHOD_APPROX_90, "#3c78d8", METHOD_APPROX_90),
            (bar_width / 2, METHOD_NATIVE_80, "#93c47d", METHOD_NATIVE_80),
        ):
            rows = subset.loc[subset["interval"] == method].set_index("regime").reindex(REGIMES)
            axis.bar(
                regime_positions + offset,
                rows["coverage"].to_numpy(dtype=float),
                width=bar_width,
                color=color,
                label=label,
            )
        axis.axhline(0.9, color="#cc0000", linestyle="--", linewidth=1, label="90% nominal")
        axis.axhline(0.8, color="#666666", linestyle=":", linewidth=1, label="80% nominal")
        axis.set_title(f"{horizon * 5} minutes")
        axis.set_xticks(regime_positions, REGIMES, rotation=15)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Empirical coverage")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("PEMS-BAY Chronos-Bolt coverage by outcome regime\nApproximate nominal 90% and separate native 80% intervals")
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_report(metrics: pd.DataFrame, config: dict, output_path: Path, metadata: dict) -> None:
    all_rows = metrics.loc[metrics["seed"] == "all"]
    lines = [
        "# Phase 1: PEMS-BAY Chronos-Bolt reproduction",
        "",
        "This is a historical reproduction, not a post-cutoff test. PEMS-BAY dates precede Chronos-Bolt publication and may overlap its training corpus.",
        "",
        f"- Period: {config['period_start']} through 2017-05-31 23:55.",
        f"- Chronological blocks: train {metadata['train_rows']:,}, calibration {metadata['calibration_rows']:,}, test {metadata['test_rows']:,} rows.",
        f"- Forecast context: {config['context_steps']} five-minute steps; horizons 3, 6, and 12 steps.",
        f"- Sensors: top {config['top_sensor_count']} ranked using training data only; {config['sensor_sample_count']} sampled for each seed {config['sensor_seeds']}.",
        f"- Test origins: {config['test_window_count']} sampled with seed {config['window_seed']} and shared across sensor groups.",
        f"- Valid candidate test origins before sampling: {metadata['valid_test_origins']:,}; windows crossing a non-5-minute timestamp gap are excluded.",
        f"- Source CSV SHA-256: `{metadata['data_sha256']}`.",
        "- Transition labels use the observed target prefix at each horizon (h=3, 6, or 12) and are created only after forecasts are cached.",
        "- Assumption: the paper does not specify how its 50 test windows were sampled; this run samples uniformly without replacement from valid chronological test origins.",
        "- Approximate nominal 90% bounds: q05 = q10 - 0.5(q20-q10); q95 = q90 + 0.5(q90-q80). This assumes local linearity of both tails and is not a native or conformal 90% interval.",
        "- Native P10-P90 results are reported separately as 80% intervals.",
        "",
        "## Coverage by horizon and regime",
        "",
        "| Horizon | Regime | Interval | N | Coverage | Mean width (mph) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, row in all_rows.iterrows():
        coverage = "n/a" if pd.isna(row.coverage) else f"{row.coverage:.3f}"
        width = "n/a" if pd.isna(row.mean_width_mph) else f"{row.mean_width_mph:.2f}"
        lines.append(
            f"| {int(row.horizon_minutes)} min | {row.regime} | {row.interval} | {int(row.n)} | {coverage} | {width} |"
        )
    headline = all_rows.loc[
        (all_rows["horizon_steps"] == 12)
        & (all_rows["regime"] == "transition")
        & (all_rows["interval"] == METHOD_APPROX_90)
    ].iloc[0]
    native = all_rows.loc[
        (all_rows["horizon_steps"] == 12)
        & (all_rows["regime"] == "transition")
        & (all_rows["interval"] == METHOD_NATIVE_80)
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"At 60 minutes on transition target periods, approximate nominal 90% coverage is {headline.coverage:.3f} (n={int(headline.n)}). The separate native 80% P10-P90 coverage is {native.coverage:.3f} (n={int(native.n)}).",
            "",
            f"The paper reports 54.9% Chronos-Bolt transition coverage for a nominal 90% interval. This run's approximate result is {headline.coverage * 100:.1f}% ({(headline.coverage - 0.549) * 100:+.1f} percentage points versus that reference). This is a descriptive gap, not a direct reproduction match: the tail extrapolation is explicit here, the paper's 50-window selection rule is unspecified, and PEMS-BAY may overlap Chronos pretraining data. No tuning was done against the published value.",
            "",
            "See `coverage_by_regime.png` for the saved plot and `coverage_by_regime.csv` for seed-level rows.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase1(config_path: Path, repo_root: Path, force: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = repo_root / config["dataset_path"]
    cache_path = repo_root / config["cache_path"]
    output_dir = repo_root / config["output_dir"]
    data_hash = _sha256(data_path)
    cache_metadata = _cache_metadata(config, data_hash)
    frame, sensor_columns = _load_period(data_path, config["period_start"], config["period_end_exclusive"])
    train_end, calibration_end = _split_indices(
        len(frame), config["train_fraction"], config["calibration_fraction"]
    )
    values = frame[sensor_columns].to_numpy(dtype=np.float32)
    groups, top_sensors = _sensor_groups(
        values[:train_end],
        sensor_columns,
        config["top_sensor_count"],
        config["sensor_sample_count"],
        config["sensor_seeds"],
        config["congested_mph"],
    )
    valid_origins = _valid_test_origins(
        frame["timestamp"],
        calibration_end,
        len(frame),
        config["context_steps"],
        max(config["horizons"]),
        config["sampling_interval_minutes"],
    )
    origins = select_window_origins(valid_origins, config["test_window_count"], config["window_seed"])

    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_metadata = json.loads(str(cache["metadata"].item()))
            if cached_metadata != cache_metadata:
                raise ValueError("forecast cache does not match this config/data; rerun with --force to replace it")
            forecasts = cache["forecasts"]
            actuals = cache["actuals"]
            cached_origins = cache["origins"]
            cached_sensors = cache["sensors"]
            if not np.array_equal(cached_origins, origins) or cached_sensors.tolist() != groups:
                raise ValueError("forecast cache sensor/window selections do not match; rerun with --force")
        cache_status = "hit"
    else:
        forecaster = ChronosBoltCPU(config["model_id"], config["torch_threads"])
        forecasts, actuals = _generate_forecasts(
            frame,
            sensor_columns,
            groups,
            origins,
            config["context_steps"],
            max(config["horizons"]),
            forecaster,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            metadata=np.array(json.dumps(cache_metadata, sort_keys=True)),
            forecasts=forecasts,
            actuals=actuals,
            origins=origins,
            sensors=np.asarray(groups),
            quantile_levels=QUANTILES,
        )
        cache_status = "created"

    metrics = pd.DataFrame(_metric_rows(forecasts, actuals, config["horizons"], config["sensor_seeds"], config))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "coverage_by_regime.csv"
    plot_path = output_dir / "coverage_by_regime.png"
    report_path = output_dir / "report.md"
    metrics.to_csv(metrics_path, index=False, float_format="%.8f")
    _plot_metrics(metrics, plot_path)
    split_metadata = {
        "train_rows": train_end,
        "calibration_rows": calibration_end - train_end,
        "test_rows": len(frame) - calibration_end,
        "valid_test_origins": len(valid_origins),
        "data_sha256": data_hash,
        "sampled_origins": origins.tolist(),
        "top_sensors": top_sensors,
        "sensor_groups": groups,
    }
    _write_report(metrics, config, report_path, split_metadata)
    return {
        "metrics": metrics,
        "cache_status": cache_status,
        "frame_rows": len(frame),
        "split_metadata": split_metadata,
        "metrics_path": metrics_path,
        "plot_path": plot_path,
        "report_path": report_path,
        "cache_path": cache_path,
    }
