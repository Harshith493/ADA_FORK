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
