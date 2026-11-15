# FORK results and reporting guide

## Headline results

All reported point estimates use the fixed PEMS-BAY test sample. Phases 0–5 did not compute confidence intervals; do not infer uncertainty for those results. Phase 6 intervals below are percentile 95% day-block bootstrap CIs (1,000 resamples, seed 6060). The Phase 6 60-minute point MAEs are over **all test windows at 60m**, not transition-only windows.

| Phase / method | Headline result | 95% CI |
|---|---|---|
| 0 · Chronos-Bolt CPU sanity forecast | One sensor/window only; no performance claim. Median warm forecast time 0.3125 process CPU seconds / 0.0857 wall seconds. | Not estimated |
| 1 · Chronos approximate nominal-90, transition 60m | See `outputs/phase1/coverage_by_regime.csv`; pooled 3-seed measured coverage is recorded in `outputs/phase1/report.md`. | Not computed |
| 2 · History-sampling, transition 60m | 73.9% vs published comparison 81–82%; gap −7.1 to −8.1 percentage points. | Not computed |
| 2 · BMA+ACI label-free global adaptation, transition 60m | 72.4% vs published BMA+ACI 73.6% (−1.2 pp; methods differ because the paper gate uses training regime labels). | Not computed |
| 3 · p(fork) detector AUPRC | 0.4299 vs volatility 0.4450 and recent-drop 0.4527. Kill/pivot signal active. | Not computed |
| 4 · FORK branches, transition 60m | About 82.1%; initial branch distribution used native P10–P90 (80%), not nominal 90%. | Not computed |
| 5 · FORK ACI (p(fork)), transition 60m, before physical clipping | 82.2%; calm/forked group coverage 88.4% / 89.3%; max-score path-region coverage about 66% in each group. | Not computed in Phase 5 |
| 6 · FORK ACI, clipped, transition 60m | Pending the current clipped replay; see `outputs/phase6/headline_60m_ci.md`. | Day-block CI in that table |
| 6 · Chronos scaled ACI, clipped, transition 60m | Pending the current clipped replay; see `outputs/phase6/headline_60m_ci.md`. | Day-block CI in that table |

The Phase 6 headline table includes transition coverage, calm/forked group coverage, transition width, interval score, and MAE with CIs. Its point MAE is explicitly labelled **all test windows at 60m**. Earlier Phase 5 point MAE (FORK 3.000 mph; Chronos q50 3.098 mph) pooled all test windows across 15/30/60m and is labelled accordingly in its CSV/markdown table.

## Matched-coverage sharpness

Sharpness at matched coverage: not established. The matched-coverage files could not be verified after clipping. In the clipped Phase 6 headline table, FORK reached higher transition coverage (82.2%, CI 79.0 to 86.3) than Chronos scaled ACI (74.3%, CI 67.0 to 82.9) at a lower mean 60-minute transition width (34.2 vs 40.7 mph). The CIs overlap, so this is suggestive, not proven. The earlier 63% figure came from unclipped Chronos intervals and is not used.

## Phase summaries (10 lines maximum each)

### Phase 0

1. Loaded cached Chronos-Bolt Small on CPU and produced one forecast plot.
2. Median forecast time after warm-up: 0.3125 CPU seconds and 0.0857 wall seconds.
3. Sanity check only; no accuracy or coverage claim.
4. Plot: `outputs/phase0/chronos_bolt_pems_bay.png`.

### Phase 1

1. Reproduced the historical PEMS-BAY setup with 168-step context and 3/6/12-step horizons.
2. Used 30 sensors per seed and 50 shared test origins; train/calibration/test splits are chronological.
3. Approximate nominal-90 used linear P05/P95 extrapolation; native P10–P90 was separate.
4. Full table and plot: `outputs/phase1/coverage_by_regime.csv`, `coverage_by_regime.png`, `report.md`.
5. No CI was computed. PEMS-BAY 2017 may overlap Chronos training data.

### Phase 2

1. Implemented ACI, history-sampling, and label-free BMA-style mixture baselines.
2. History-sampling transition coverage: 73.9% vs paper's approximate 81–82%.
3. Label-free global BMA+ACI transition coverage: 72.4%; not exact paper BMA, whose gate uses training regime labels.
4. Consecutive-origin ACI: 58.2% transition, 91.7% calm, width 18.39 mph.
5. Point MAE in the headline table is all test windows at 60m; no CI was computed.
6. Outputs: `outputs/phase2/report.md`, `metrics_by_regime.csv`, calibration sweep CSVs.

### Phase 3

1. Detector predicts Chronos approximate-90 misses; no regime labels enter fitting/tuning.
2. p(fork) transition AUPRC 0.4299, below volatility 0.4450 and recent-drop 0.4527.
3. Therefore the detector kill/pivot signal is active; call p(fork) a risk score, not calibrated fork probability.
4. Outputs: `outputs/phase3/detector_metrics.csv`, `precision_recall.csv`, `synthetic_metrics.csv`, `report.md`.
5. No CI was computed.

### Phase 4

1. Built at most two neighbor branches with temporal dependence; no sensor coupling.
2. Headline branch transition coverage was about 82.1%, with 90.7% calm coverage and 16.83 mph mean width.
3. Initial Phase 4 intervals used P10–P90 (80%); Phase 5/6 calibrate to 90%, so coverage comparisons need this caveat.
4. Outputs: `outputs/phase4/headline_60m.csv`, `results.csv`, `D8_verdict.md`, `example_branches.png`.
5. No CI was computed.

### Phase 5

1. Online split/path conformal states are pooled by gate group; outcomes update only at maturity.
2. p(fork) group interval coverage was 88.4% calm / 89.3% forked; transition coverage was 82.2%.
3. Max-score path-region coverage was ~66%; the nonconformity cap and finite branch support prevent expansion beyond observed samples.
4. Joint coverage of 12 marginal intervals is a separate, stricter metric.
5. Calm-collapse coverage was 82.9%; gamma selected at grid minimum .001.
6. Corrected uncapped Chronos score is `|y-q50| / max((q90-q10)/2, 1e-3)`; old capped Chronos rows are invalid baselines.
7. Outputs: `outputs/phase5/headline_60m.csv`, `whole_path_coverage.csv`, `cumulative_coverage.png`.

### Phase 6

1. Clipped interval endpoints to each sensor's training min/max before width, interval score, and coverage; see final A1 before/after note in `HANDOVER.md`.
2. Reran calibration-only gamma tuning for alpha .3/.2/.1/.05, then test matched-coverage curves at 15/30/60m.
3. Matched calm means free-flow plus congested; gate-group coverage is reported separately.
4. Branch odds: Brier 0.2141; in predicted minority weights [.25,.35], nearest-minority frequency 33.1% (n=1,157).
5. 1,000 day-block bootstrap resamples use seed 6060.
6. Outputs: `outputs/phase6/headline_60m_ci.md`, `matched_coverage.csv`, `matched_calm_coverage.csv`, `branch_reliability.csv`, `day_block_bootstrap_ci.csv`.
7. Scale-up and METR-LA transfer were not run; see `HANDOVER.md`.

## Baselines and published comparison points

| Baseline | Published comparison | Our result | Interpretation |
|---|---:|---:|---|
| Native Chronos-Bolt / approximate nominal-90 | 54.9% transition coverage | Phase 1 measured value in `outputs/phase1/report.md`; Phase 5's sampled native control 49.7% | Historic reproduction; no CI, not post-cutoff evidence |
| History-sampling | About 81–82% transition coverage | 73.9% | 7.1–8.1 pp lower; our implementation guesses documented in Phase 2 notes |
| BMA + ACI | 73.6% | 72.4% for our label-free global adaptation | Not exact BMA; paper gate uses training-only regime labels |
| ACI | No headline published point in project brief | Consecutive-origin 58.2% transition, 91.7% calm, 18.39 mph | ACI reference and Phase 5 engine |
| Copula coupling | Reference methods only | Phase 4 tested Chronos global copula; see `outputs/phase4/headline_60m.csv` | Local implementation; no reference code reused |

## Decisions, deviations, limitations

- User-approved configurations and autonomous choices: `DECISIONS.md`; run details: `EXPERIMENTS.md`; reuse inventory: `CREDITS.md`.
- Deviations: Phase 2 BMA is label-free and not exact BMA; Phase 4 branches start from an 80% native Chronos band; no true-regime branch-construction oracle, sensor coupling, other TSFMs, METR-LA transfer, Caltrans 2026, fev-bench, or GIFT-Eval.
- Phase 3 p(fork) did not beat volatility or recent-drop AUPRC; the active kill signal remains.
- FORK per-step group coverage is near 90%, but the whole-path region is only ~66% due finite branch support. Do not claim a 90% path guarantee.
- The 82.2% transition point estimate misses 85%, even though its Phase 6 CI spans 85%; do not claim the target was reached.
- PEMS-BAY 2017 and METR-LA 2012 are historical; Chronos traffic-data overlap is unresolved. No post-cutoff generalization claim is supported.
- There is no confidence interval for Phases 0–5. Phase 6 bootstrap CIs apply only to the fixed sampled evaluation, not all test origins.

## Claims allowed / claims not allowed

**Allowed:** report the measured fixed-sample Phase 6 clipped estimates and their day-block CIs; say that FORK reached higher transition coverage (82.2%) than Chronos scaled ACI (74.3%) at a lower mean 60-minute transition width (34.2 vs 40.7 mph) and a lower interval score (57.9 vs 109.5), noting that the coverage and width CIs overlap, so this is suggestive, not proven; say that sharpness at matched coverage is not established because the matched-coverage files could not be verified after clipping; report the 30% branch-odds bin result and Brier score.

**Not allowed:** call p(fork) a calibrated fork probability; claim transition coverage reaches 85% based only on a CI spanning 85%; claim a 90% whole-path guarantee; call our BMA adaptation exact BMA; claim post-cutoff validation, cross-sensor coupling, multi-model transfer, METR-LA/Caltrans/fev-bench/GIFT-Eval results, or an oracle branch construction result.
