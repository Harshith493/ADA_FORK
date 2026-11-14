# Phase 1: PEMS-BAY Chronos-Bolt reproduction

This is a historical reproduction, not a post-cutoff test. PEMS-BAY dates precede Chronos-Bolt publication and may overlap its training corpus.

- Period: 2017-01-01 00:00:00 through 2017-05-31 23:55.
- Chronological blocks: train 30,433, calibration 4,347, test 8,696 rows.
- Forecast context: 168 five-minute steps; horizons 3, 6, and 12 steps.
- Sensors: top 50 ranked using training data only; 30 sampled for each seed [42, 43, 44].
- Test origins: 50 sampled with seed 42 and shared across sensor groups.
- Valid candidate test origins before sampling: 8,685; windows crossing a non-5-minute timestamp gap are excluded.
- Source CSV SHA-256: `c39dc8f5f3a7f1e9e9deeab1f3fdfa7b87d2b4e45b4075d4c3bd3c62c3051ef2`.
- Transition labels use the observed target prefix at each horizon (h=3, 6, or 12) and are created only after forecasts are cached.
- Assumption: the paper does not specify how its 50 test windows were sampled; this run samples uniformly without replacement from valid chronological test origins.
- Approximate nominal 90% bounds: q05 = q10 - 0.5(q20-q10); q95 = q90 + 0.5(q90-q80). This assumes local linearity of both tails and is not a native or conformal 90% interval.
- Native P10-P90 results are reported separately as 80% intervals.

## Coverage by horizon and regime

| Horizon | Regime | Interval | N | Coverage | Mean width (mph) |
|---|---|---|---:|---:|---:|
| 15 min | free-flow | approximate nominal 90% | 3626 | 0.892 | 5.94 |
| 15 min | transition | approximate nominal 90% | 445 | 0.656 | 15.49 |
| 15 min | congested | approximate nominal 90% | 429 | 0.946 | 15.46 |
| 15 min | free-flow | native 80% (P10-P90) | 3626 | 0.842 | 4.82 |
| 15 min | transition | native 80% (P10-P90) | 445 | 0.600 | 13.02 |
| 15 min | congested | native 80% (P10-P90) | 429 | 0.911 | 12.70 |
| 30 min | free-flow | approximate nominal 90% | 3547 | 0.877 | 7.52 |
| 30 min | transition | approximate nominal 90% | 575 | 0.588 | 18.54 |
| 30 min | congested | approximate nominal 90% | 378 | 0.950 | 20.57 |
| 30 min | free-flow | native 80% (P10-P90) | 3547 | 0.830 | 6.11 |
| 30 min | transition | native 80% (P10-P90) | 575 | 0.527 | 15.49 |
| 30 min | congested | native 80% (P10-P90) | 378 | 0.923 | 16.65 |
| 60 min | free-flow | approximate nominal 90% | 3417 | 0.872 | 10.31 |
| 60 min | transition | approximate nominal 90% | 782 | 0.497 | 21.98 |
| 60 min | congested | approximate nominal 90% | 301 | 0.927 | 28.27 |
| 60 min | free-flow | native 80% (P10-P90) | 3417 | 0.816 | 8.36 |
| 60 min | transition | native 80% (P10-P90) | 782 | 0.439 | 18.26 |
| 60 min | congested | native 80% (P10-P90) | 301 | 0.897 | 22.88 |

## Headline

At 60 minutes on transition target periods, approximate nominal 90% coverage is 0.497 (n=782). The separate native 80% P10-P90 coverage is 0.439 (n=782).

The paper reports 54.9% Chronos-Bolt transition coverage for a nominal 90% interval. This run's approximate result is 49.7% (-5.2 percentage points versus that reference). This is a descriptive gap, not a direct reproduction match: the tail extrapolation is explicit here, the paper's 50-window selection rule is unspecified, and PEMS-BAY may overlap Chronos pretraining data. No tuning was done against the published value.

See `coverage_by_regime.png` for the saved plot and `coverage_by_regime.csv` for seed-level rows.
