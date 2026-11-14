# Experiments

## Phase 0 inference sanity check

- Purpose: confirm that the public Chronos-Bolt Small weights can produce a CPU forecast on PEMS-BAY and render a plot. This is not a performance evaluation.
- Command: `python scripts\phase0_forecast.py`
- Model: `amazon/chronos-bolt-small`; Chronos package 2.3.2; CPU.
- Data: Zenodo PEMS-BAY CSV, sensor column `400001`; no filtering or label use.
- Forecast point: index `int(49679 * 0.70)` in file order; 168 prior points (14h); 12 forecast points (1h); quantiles 0.1, 0.5, 0.9.
- Output: `outputs/phase0/chronos_bolt_pems_bay.png`.
- Observed output after checking the quantile columns: first median forecast 67.66 mph; first observed value 68.10 mph. Single-point descriptive check only.
- Timing: one warm-up, then 10 repeats on the same already-loaded model and input; median process CPU time 0.3125 s per forecast and median elapsed wall time 0.0857 s per forecast. Small-sample timing, not a performance benchmark.
- Reproducibility: source data is in `data/raw/PEMS-BAY.csv`; model cache in ignored `.cache/huggingface`. Seed is not applicable to the deterministic point forecast.
- Native Chronos-Bolt quantiles are 0.1, 0.2, ..., 0.9. P10-P90 is an 80% interval. The API clips requested quantiles outside the trained range, so P05/P95 requests do not produce a native 90% interval.
- Known limitation: this script does not implement a 90% band or coverage metric.

## Phase 1: frozen Chronos-Bolt reproduction

- Approved choices: 1A, 2A, 3A, 4B, 5A. Full method, rationale, and tail assumption are in `DECISIONS.md`.
- One-command run: `python scripts\run_phase1.py`; the forecast cache is reused when data and config hashes match. Use `python scripts\run_phase1.py --force` only to replace the cache and recompute.
- Config: `configs/phase1_pems_bay.json`.
- Data: PEMS-BAY rows from 2017-01-01 through 2017-05-31 inclusive; source SHA-256 `c39dc8f5f3a7f1e9e9deeab1f3fdfa7b87d2b4e45b4075d4c3bd3c62c3051ef2`. The source has a 65-minute timestamp gap; context/target windows crossing a non-5-minute gap are excluded.
- Split: chronological row blocks, 30,433 train / 4,347 calibration / 8,696 test rows. Sensor ranking uses only training rows. Calibration is reserved and not used by the frozen quantile extrapolation.
- Sampling: top 50 training-congested sensors; 30 sensors for seeds 42, 43, and 44; 50 valid test origins selected once using seed 42 and shared across sensor groups. The paper does not say how it sampled 50 windows; uniform random selection without replacement is an explicit implementation assumption.
- Forecasts: 168-step context, 12-step Chronos-Bolt Small CPU forecasts, batch 30 sensors per origin; cache contains 5 native quantiles at every forecast step for all 4,500 sensor-window-seed combinations.
- Regime labels are calculated from the observed future prefix at each horizon after predictions are cached. The forecaster API receives only context arrays.
- Intervals: approximate nominal 90% via q05/q95 linear tail extrapolation; native P10-P90 reported separately as 80%. Do not combine rows.
- Measured headline at 60 minutes / transition: approximate nominal 90% coverage 49.7% (n=782); native 80% coverage 43.9% (n=782). Published reference [1] is 54.9% for its nominal-90 result; this run is 5.2 percentage points lower, and is not an exact protocol reproduction because the interval-tail method and window sampling differ.
- All horizons and regimes, seed-level rows, counts, and mean widths are in `outputs/phase1/coverage_by_regime.csv`; plot and summary are `outputs/phase1/coverage_by_regime.png` and `outputs/phase1/report.md`.
- Verification requested by the user: `python -m unittest discover -s tests -v` passes 8 tests (coverage metric, regime labels, interval extrapolation, and label isolation). A second run returned `cache=hit` and identical headline metrics.
- No tuning was performed against the published result. PEMS-BAY is 2017 and may overlap Chronos training data; this is a historical reproduction, not a post-cutoff test.
