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

## Phase 2: ACI, history sampling, and label-free BMA-style adaptation

### Phase 2 audit and consecutive-update ACI variant

- History futures are drawn only from valid training-block origins whose complete 168-step context and 12-step future lie before the training boundary. Calibration and test origins are later blocks, so those neighbor windows cannot overlap them. Windows within the training pool can overlap one another.
- Sample-based MAE is computed from the per-horizon median of the selected neighbors' future paths; BMA-style sample MAE uses the median of the 100-path mixed sample. Chronos/ACI point MAE uses q50.
- Approximate-90 ACI selected gamma 0.001, the minimum of [0.001, 0.005, 0.01, 0.02]. The original sampled-window ACI updated only on the 50 sampled test origins per sensor.
- Added a separate approximate-90 ACI variant that forecasts every timestamp-contiguous valid test origin and updates from all matured 15/30/60-minute miss events. It is scored on the same sampled origins as every other method. Pilot: 30 batches in 2.9s; projected full pass 41.4 minutes; completed under 2 hours. Cached sampled-origin bounds: data/cache/phase2_consecutive_aci_sampled_bounds.npz.
- Consecutive-update headline (60m transition): coverage 58.2% (n=782), calm coverage 91.7% (n=3,718), mean width 18.39 mph, interval score 48.65, point MAE 4.45. This improves coverage but widens intervals; it uses the same calibration-selected gamma and is not a new gamma tune.
- Audit of Phase 2 selections found no use of test outcomes or labels for tuning. Existing DECISIONS.md had no “Autonomous choices” section; the Phase 3 autonomous choices are added below.

- Re-verified reference [1] and [7] from their paper records/full text. No official code for [1] was found; no external code was reused.
- One-command run: `python scripts\run_phase2.py`; use `--force` only to replace the separate calibration forecast cache. Phase 1 test cache is always reused.
- Config: `configs/phase2_pems_bay.json`. Calibration cache: `data/cache/phase2_calib_forecasts.npz` (50 origins x 3 x 30 sensors); test cache shape `(3, 30, 50, 12, 5)`.
- kNN timing pilot: 20 queries in 0.01-0.02 seconds; linear estimate for matching the distinct selected sensors/origins was about 0.1 minutes. Whole run completed in roughly one minute after model load. No hosted runtime was needed.
- Exact matching uses 168-point train-standardized sensor contexts and timestamp-contiguous windows only. Train block remains strictly before calibration/test; k selected on calibration. Gamma and weights are also chosen on calibration only. ACI states are one per sensor/method; the same origin uses one alpha across horizons, and matured outcomes update the state before a subsequent origin.
- Selected parameters: `k=50`, global weight `0.5`, paper-style 10-window weight `0.1`; gamma `0.001` approximate90, `0.005` native80, `0.001` each BMA+ACI variation. Full sweeps in `outputs/phase2/`.
- Headline 60-minute transition coverage: native approximate90 49.7%; ACI approximate90 50.4%; history sampler 73.9%; label-free BMA+ACI paper-style 54.9%; label-free BMA+ACI global 72.4%. Native P10-P90 43.9%; its ACI version 44.9%. See `outputs/phase2/headline.csv` for widths, interval scores, MAE and calm coverage.
- History sampler's 73.9% is 7.1-8.1 points below [1]'s approximate 81-82%. Global label-free BMA adaptation is 1.2 points below [1]'s 73.6%, but is not exact BMA (the paper gate uses training-only destination-regime labels). Paper-style adaptation gives 54.9%. These are descriptive, not like-for-like reproduction gaps.
- Tests: `python -m unittest discover -s tests -v` passes 15 tests, including ACI update, interval score, history quantiles, mixture replacement, train-only matching and context-only forecaster boundary.
- Implementation guesses and deviations are enumerated in `DECISIONS.md` and `outputs/phase2/report.md`; no external baseline code was reused.

## Phase 3: fork detector (PEMS-BAY and synthetic)

- One-command run: `python scripts\run_phase3.py`; config: `configs/phase3_pems_bay.json`. Tests: `python -m unittest discover -s tests -v`.
- No additional dependencies were installed. The project does not contain a requirements.txt; k-means and logistic regression are implemented with NumPy.
- Phase 2 concern audit: history future candidates were training-only and ended before calibration; they cannot overlap the later cal/test blocks. Sample-based MAE uses per-step sample median. Approximate-90 gamma=0.001 was at the sweep minimum; original ACI updated sampled origins only. Added all-valid-origin ACI (same cal gamma), which produced 58.2% transition coverage, 91.7% calm coverage, 18.39 mph mean width, 48.65 interval score, and 4.45 MAE.
- Training forecast cache uses 5,000 seeded windows from 180-row spacing per sensor; each full context and target stays in train. Train/cal/test row blocks remain 30,433/4,347/8,696. The matcher excludes +/-180 rows for train self-matches and candidate histories remain train-only.
- Phase 3 features: normalized recent 168 readings plus cyclic time/day features in matching; two-means separation, minority share, Chronos approximate-90 spread, and neighbor/Chronos median gap. Fit target is the Chronos approximate-90 miss indicator, never transition labels. Cal tunes k/L2 and thresholds; test transition labels enter only scoring.
- Pilot: 20 kNN+2-means feature windows in 0.08s, projected 0.9 min for feature extraction. Prior Chronos throughput projected 0.3 min for 5,000 forecasts; combined projection about 1.2 minutes. Run completed with caches created.
- Selected k=25, L2=0.01; calibration miss fraction 15.96%; hard-flag threshold 0.168745; test flag rate 18.67%.
- Test AUPRC: p(fork) 0.4291; volatility 0.4438; recent-drop 0.4527; CUSUM 0.3405. p(fork) is below volatility, activating the kill/pivot signal. Calm false-alarm rates: 0.123, 0.874, 0.186, 1.000 respectively.
- Synthetic: 600 episodes, two speed regimes, 24 mph jump, branch coupling 0.8. Known mean odds 0.442; mean p 0.609; odds-bin MAE 0.310; Brier vs odds 0.239; AP vs realized fork events 0.387 (event prevalence 0.471). Probability transfer is poor.
- Full outputs: `outputs/phase3/report.md`, `detector_metrics.csv`, `calibration_sweep.csv`, `precision_recall.csv`, `test_predictions.csv`, synthetic CSVs and plots. Phase 4/5 cache: `data/cache/phase3_fork_outputs.npz` (ignored local file), containing calibration/test p(fork) and minority shares.
- All 25 unit tests pass, including label-dependency, no-self-match, train-only neighbor, toy two-means, tie-safe PR, and fixed-seed determinism checks. No external code or new package was reused.
