# Experiments

## Phase 0 inference sanity check

- Purpose: confirm that the public Chronos-Bolt Small weights can produce a CPU forecast on PEMS-BAY and render a plot. This is not a performance evaluation.
- Command: `python scripts\phase0_forecast.py`
- Model: `amazon/chronos-bolt-small`; Chronos package 2.3.2; CPU.
- Data: Zenodo PEMS-BAY CSV, sensor column `400001`; no filtering or label use.
- Forecast point: index `int(49679 * 0.70)` in file order; 168 prior points (14h); 12 forecast points (1h); quantiles 0.1, 0.5, 0.9.
- Output: `outputs/phase0/chronos_bolt_pems_bay.png`.
- Observed output: first median forecast 68.92 mph; first observed value 68.10 mph. Single-point descriptive check only.
- Reproducibility: source data is in `data/raw/PEMS-BAY.csv`; model cache in ignored `.cache/huggingface`. Seed is not applicable to the deterministic point forecast.
- Known limitation: P10-P90 is an 80% band. This script does not implement a 90% band or coverage metric.
