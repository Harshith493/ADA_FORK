# Phase 2: PEMS-BAY baselines

All methods share Phase 1 cached test forecasts, sensors and chronological test origins. Regime labels are generated only after predictions and used only for scoring. Calibration tuning uses the preceding chronological block; online ACI updates only after outcomes mature before a later forecast origin. PEMS-BAY (2017) may overlap Chronos training data, so this is a historical reproduction.

- Calibration forecast cache status: hit; 50 sampled calibration origins. Test cache reused unchanged.
- kNN timing pilot: 20 exact queries in 0.012s; projected matching 2.9s. The initial Phase 2 calibration/matching run completed in about one minute after model load.
- Consecutive-origin ACI: pilot of 30 Chronos batches took 2.9s; projected full test forecast pass was 41.4 minutes. The pass completed under the two-hour cap; selected-origin bounds are cached in data/cache/phase2_consecutive_aci_sampled_bounds.npz.
- Calibration selections: k=50, paper-style weight=0.10, global weight=0.50. Gamma selections appear in the sweep table.
- Point forecast is the distribution median; ACI changes bounds but not the point forecast. Calm pools free-flow and congested; mean width and interval score pool all regimes at 60 minutes.
- ACI width mapping is an implementation assumption: multiply the symmetric base radius by sqrt(target_alpha/current_alpha), so a smaller online miscoverage level widens the interval. [1] gives no gamma for this experiment.
- Chronos pseudo-samples use deterministic inverse-CDF linear interpolation over q10/q20/q50/q80/q90; tails are held at q10/q90. Mixtures replace round(100*w) model paths with sampled neighbor paths.
- [1] BMA gate uses training-only destination-regime probabilities. This implementation has no regime categories or predicted-regime gate and is explicitly not exact BMA from [1].

## Headline at 60 minutes

| Method | Transition coverage (n) | Calm coverage (n) | Mean width | Interval score | Point MAE |
|---|---:|---:|---:|---:|---:|
| ACI approximate nominal 90% | 0.504 (782) | 0.883 (3718) | 13.84 | 48.20 | 4.45 |
| ACI approximate nominal 90% (updates every valid step) | 0.582 (782) | 0.917 (3718) | 18.39 | 48.65 | 4.45 |
| ACI native 80% (P10-P90) | 0.449 (782) | 0.819 (3718) | 11.19 | 30.67 | 4.45 |
| BMA+ACI global weight — our label-free adaptation of BMA, not exact BMA from [1] | 0.724 (782) | 0.896 (3718) | 15.04 | 27.69 | 3.71 |
| BMA+ACI paper-style weight — our label-free adaptation of BMA, not exact BMA from [1] | 0.549 (782) | 0.847 (3718) | 12.29 | 40.62 | 4.28 |
| History-sampling (our version of [1]) | 0.739 (782) | 0.814 (3718) | 11.61 | 24.35 | 3.48 |
| Native Chronos-Bolt (approximate nominal 90%) | 0.497 (782) | 0.876 (3718) | 13.54 | 48.35 | 4.45 |
| Native Chronos-Bolt (native 80%) | 0.439 (782) | 0.822 (3718) | 11.05 | 30.67 | 4.45 |

## Calibration gamma sweep (mean sequential interval score)

| Method | 0.001 | 0.005 | 0.01 | 0.02 | Selected |
|---|---:|---:|---:|---:|---:|
| ACI approximate nominal 90% | 29.70 | 33.20 | 37.45 | 48.31 | 0.001 |
| ACI native 80% (P10-P90) | 18.54 | 18.44 | 18.49 | 18.96 | 0.005 |
| BMA+ACI paper-style weight — our label-free adaptation of BMA, not exact BMA from [1] | 24.34 | 25.43 | 26.20 | 27.67 | 0.001 |
| BMA+ACI global weight — our label-free adaptation of BMA, not exact BMA from [1] | 17.18 | 17.27 | 17.65 | 18.40 | 0.001 |

## Published comparison (descriptive, not protocol matched)

- History-sampling: ours 73.9% versus [1] about 81–82%, a gap of -7.1 to -8.1 percentage points. [1] does not specify its conditioning estimator or matching details.
- BMA + ACI: [1] reports 73.6%. Our global label-free adaptation is 72.4% and the paper-style 10-window weight adaptation is 54.9%; neither is exact BMA because the [1] training-only regime gate is omitted.
- Native Chronos-Bolt approximate nominal-90 result is 49.7% versus [1] 54.9%, a descriptive gap of -5.2 points; our tail extrapolation and window sampling are explicit approximations.

## Guesses and deviations

- History sampler: per-sensor train-only z-scored 168-step contexts, exact Euclidean matching, timestamp-contiguous training origins, k in [5, 10, 20, 50], actual next-hour neighbor paths, and empirical 5th/95th percentile interval. [1] describes current-speed conditioning and 100 draws but omits this matching rule, k, and interval construction.
- Paper-style weight: first sensor in seed-42 group, first 10 sampled calibration origins, and mean 60-minute interval score objective. [1] says 10 held-out windows but leaves sensor/window selection and objective unspecified. Global weight pools all calibration sensors/windows.
- Gamma candidates are [0.001, 0.005, 0.01, 0.02], selected by calibration-only sequential interval score. The approximate-90 choice 0.001 is the lower edge of that grid. The original ACI row updates only at sampled test origins; the extra ACI row forecasts each valid consecutive 5-minute origin, with a state shared across horizons and only matured outcomes updating it.
- Pseudo-sample interpolation/tail handling and ACI width mapping above are implementation assumptions. Full rows across 15/30/60 minutes, regimes, counts, widths, interval scores and MAE: `metrics_by_regime.csv`. k/weight/gamma sweeps and selected values are adjacent CSVs.
- Plots: `coverage_by_regime.png`, `width_vs_coverage.png`. No official baseline code was found or reused; all baseline logic is implemented locally.
