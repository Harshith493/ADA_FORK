# HANDOVER

## Project goal (5 lines)

1. Implement FORK as a training-free wrapper around frozen time-series foundation models.
2. Detect likely forks from similar past windows and the model's spread.
3. Build weighted scenario paths with dependence within branches.
4. Keep interval and path coverage with online conformal calibration and separate calm/forked states.
5. Reproduce the published traffic failure, evaluate honestly, and deliver a demo and report.

Project brief: [`docs/onepager_summary.md`](docs/onepager_summary.md).

## Current status

- Phase: 3 - fork detector.
- Step: Phase 3 detector, scoring, synthetic check, report, plots, and tests complete.
- Status: Phase 3 done; awaiting user's review. Do not start Phase 4 yet.

## Completed (commit references)

- Project one-pager and initial handover: `0111ceb64dd664f9a79291f74da6267977594e75`.
- Phase 0 notes, forecast script, and plot: `3984bbe` (PEMS-BAY CSV and model cache remain local and untracked).
- Phase 0 approval, measured timing, and Phase 1 decision menu: `b2d23c3`.
- Phase 1 option refinement for the paper's Jan-May data period and scoring labels: `e4bb948`.
- Phase 1 code, config, tests, metrics, plot, and report: `538128e` (forecast cache remains local and ignored).
- Phase 2 source review, assumptions, runtime estimate, and no-reuse record: `733e73c`.
- Phase 2 implementation, tests, metrics, report, plots, and research notes: `0c7255d974dddad3daca7c686b515308d62974a1`.
- Phase 2 handover update with continuation details: `ea95679626fd77e28a5b7c648e2563e5b09c2930`.
- Phase 2 audit and consecutive-update ACI variant: `316a20d1585c6a7d46e0702e6a6f2a0199c5c6ae`.
- Phase 3 detector implementation, tests, metrics, report, figures, and research notes: `4511e501f7deb03f666d04121ddd41802486c8c1`.
- Phase 3 handover update with review state and continuation steps: `ec4b17b8b495815e4487074951d56629537e89f0`.

## Exactly what to do next

1. Review `outputs/phase3/report.md`, `outputs/phase3/precision_recall.png`, `outputs/phase3/example_neighbor_branches.png`, and `outputs/phase3/synthetic_odds.png`.
2. Review the Phase 2 audit and Phase 3 autonomous choices/results in `DECISIONS.md`.
3. Approve Phase 3 or send corrections. Do not start Phase 4 before approval.

## Decisions

- User decisions: Phase 0 approved; Phase 1 choices 1A, 2A, 3A, 4B, 5A; Phase 2 D1-D7 recorded in [`DECISIONS.md`](DECISIONS.md).
- Phase 2 is approved. Phase 3 awaits user review. Its user and autonomous choices are recorded in [`DECISIONS.md`](DECISIONS.md); no unresolved method choice remains.

## Environment and commands

- Host used for Phase 0: Windows 11, Python 3.12.6, Intel i7-13700H / 14 cores / 16 GB RAM; no NVIDIA GPU.
- Phase 0 installed `chronos-forecasting` 2.3.2 in the host Python. For a clean project environment, recommended commands:

  ```powershell
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install chronos-forecasting pandas matplotlib
  ```

- Chronos weights are cached under `.cache/huggingface` on the project drive because the C: user cache had only 44.7 MB free. The cache is excluded from Git.
- Phase 0 one-forecast command: `python scripts\phase0_forecast.py` (run from repo root; internet needed only on first model load).
- Timing: 10 repeats after one warm-up; median 0.3125 process CPU seconds and 0.0857 wall seconds per forecast with model loaded. Process CPU time can exceed wall time when inference uses multiple threads.
- Paper-sample estimate: 3 seeded subsets x 30 sensors x 50 windows = 4,500 12-step forecasts; about 23.4 aggregate CPU-minutes and 6.4 serial wall-minutes for inference, with 10-20 minutes estimated end-to-end including loading, data handling, caching, and scoring. Assumes one 12-step forecast supplies horizons 3/6/12; other choices change this estimate.
- Phase 1 command: `python scripts\run_phase1.py`; cached reruns use the existing forecasts. Cache recompute: `python scripts\run_phase1.py --force`; tests: `python -m unittest discover -s tests -v`.
- Phase 2 command: `python scripts\run_phase2.py`; tests: `python -m unittest discover -s tests -v`. Add `--force` only to recreate the separate calibration forecast cache; Phase 1 test forecasts are never recomputed.
- Calibration cache is local at `data/cache/phase2_calib_forecasts.npz`; Phase 1 test cache at `data/cache/phase1_pems_bay_forecasts.npz`. kNN pilot took about 0.01 seconds for 20 queries and projected the full matching pass below one minute. Full cache-hit run completed in about 30 seconds; calibration-forecast creation also stayed within about one minute.
- Phase 2 all-step ACI rerun: `python scripts\run_phase2.py`. Pilot 30 Chronos batches/2.9s, projection 41.4 minutes; completed and cached selected-origin bounds at `data/cache/phase2_consecutive_aci_sampled_bounds.npz`. No extra packages were installed. No `requirements.txt` is present in this repository.
- Phase 3 command: `python scripts\run_phase3.py`; use the environment already used for Phase 1/2 and the cached Chronos weights. Add no packages; implement small regression and k-means with NumPy.
- Phase 3 run completed under the two-hour limit. Pilot: 20 matching/k-means queries in 0.08s; projected features 0.9 minutes, 5,000 forecasts estimated 0.3 minutes from measured Phase 2 throughput; combined projection 1.2 minutes. Forecast, feature, and p(fork)/minority caches are local and ignored.

## File map

- `AGENTS.md` - short project rules for future agents.
- `HANDOVER.md` - current progress and exact continuation steps.
- `Aporia_One-Pager_FORK.pdf` - original project brief.
- `docs/onepager_summary.md` - compact brief and confirmed scope.
- `DECISIONS.md` - user-approved design choices and open questions.
- `EXPERIMENTS.md` - reproducible experiment configurations and outcomes.
- `CREDITS.md` - external components, sources, and licenses.
- `scripts/phase0_forecast.py` - one CPU-only Chronos-Bolt sanity forecast.
- `scripts/run_phase1.py` - one-command Phase 1 experiment; uses cached forecasts when config and data hashes match.
- `configs/phase1_pems_bay.json` - fixed Phase 1 model, time range, splits, seeds, and sampling settings.
- `src/fork/phase1.py` - scoring-only regime labels, coverage metrics, interval extrapolation, safe forecasting boundary, caching, and report generation.
- `tests/test_phase1.py` - coverage, transition label, tail extrapolation, and label-isolation tests.
- `outputs/phase1/` - all horizon/regime rows, plot, and concise report.
- `data/cache/phase1_pems_bay_forecasts.npz` - local compressed forecasts (ignored by Git).
- `configs/phase2_pems_bay.json` - Phase 2 seeds, grids, and cache/output paths.
- `scripts/run_phase2.py` - one-command calibration tuning, online test evaluation, outputs and report.
- `src/fork/window_matching.py` - train-only standardized exact window matching and empirical intervals.
- `src/fork/phase2.py` - ACI update/interval score, pseudo-sample and label-free mixture helpers.
- `tests/test_phase2.py` - ACI, score, matching, mixture, quantile and context-only boundary tests.
- `outputs/phase2/` - Phase 2 metrics, report, plots, and tuning sweep tables.
- `src/fork/phase3.py` - label-free calendar embedding, branch features, baseline scores, NumPy logistic detector, PR metrics, synthetic generator.
- `scripts/run_phase3.py`, `configs/phase3_pems_bay.json` - one-command Phase 3 pipeline and fixed settings.
- `tests/test_phase3.py` - no-label dependency, no-self-match, train-only, toy k-means, PR ties, and determinism tests.
- `outputs/phase3/` - test metrics, calibration sweep, PR curve/table, example neighbor branches, synthetic odds comparison, and report.
- `data/cache/phase3_train_5000_forecasts.npz`, `phase3_feature_cache.npz`, `phase3_fork_outputs.npz` - ignored local caches; final cache has cal/test p(fork) and minority shares for Phase 4/5.
- `data/cache/phase2_consecutive_aci_sampled_bounds.npz` - sampled-origin bounds after all-valid-step ACI updates; ignored local cache.
- Phase 3 implementation, config, tests and outputs: not yet created.
- `data/raw/` - downloaded raw public data (not committed).
- `outputs/phase0/` - Phase 0 forecast figure.
- `.cache/huggingface/` - local model cache (not committed).

## Repository structure

- `src/fork/` - FORK components, separated from regime scoring code.
- `configs/` - fixed experiment settings and seeds.
- `tests/` - metric, conformal update, and label-isolation checks.
- `scripts/` - one command per experiment and demo entry points.
- `data/raw/`, `data/processed/` - source data and cached derivatives, both untracked.
- `outputs/`, `logs/`, `reports/`, `demo/` - figures/results, run logs, final write-up, and demo assets.
- Keep project rules and handover at root, plus `docs/` for the one-pager summary and research notes.

## Data status

- PEMS-BAY: present at `data/raw/PEMS-BAY.csv` (85,718,439 bytes, Zenodo record 5146275); Phase 1 used 2017-01-01 through 2017-05-31 inclusive.
- METR-LA: not downloaded by user request; public at the same Zenodo record, defer until needed.
- Synthetic two-regime generator: implemented and scored in Phase 3 (not a separate data file).
- Phase 3 synthetic two-regime/fork generator is implemented in `src/fork/phase3.py` and evaluated in `outputs/phase3/synthetic_metrics.csv`; standalone reusable data cache is not needed.
- No manual account/token is needed for the above public files. Caltrans PeMS 2026 requires the user's manual account/download step and remains stretch-only.
- Data citation/license metadata on the Zenodo record is incomplete; do not redistribute the raw data without checking source terms.
- Chronos model training overlap with PEMS-BAY/METR-LA is not resolved by the model card. Treat the historic datasets as replication only, not post-cutoff headline evidence.

## Known issues / failed attempts / do not retry

- First model load tried the C: Hugging Face cache and failed for lack of disk space. It is fixed by setting `HF_HOME` to the project-local cache; do not restore the default cache on this machine.
- First script version used the wrong `predict_quantiles` argument name; fixed to `inputs=`.
- The model's native quantile grid spans 0.1-0.9. Phase 0 plot shows its 80% P10-P90 band; it is not evidence for 90% interval coverage.
- Requests outside 0.1-0.9 through the Chronos API clip to the outer trained levels; direct P05/P95 requests do not create a native 90% interval.
- Phase 1's approximate q05/q95 are linear extrapolations from adjacent outer deciles; this assumes local tail linearity and has no 90% guarantee.
- The sampled-window selection method is unspecified in reference [1]; the run uses one seed-42 random sample shared across the three sensor groups.
- PEMS-BAY has a single 65-minute timestamp gap within the chosen period; windows crossing non-5-minute gaps are excluded.
- The 60-minute approximate nominal-90 transition coverage is below the published 54.9%; do not tune against the reference.
- The package and model were obtained from the official Amazon Chronos distribution; no code from reference [1] or another baseline was reused.
- Eight requested unit tests pass, including a test that the forecaster receives context only.
- Reference [1] BMA explicitly uses training-estimated probabilities for destination regimes. User approved only a label-free adaptation; it is not exact BMA from [1]. No official code was found or reused.
- First Phase 2 forecast attempt failed because the runner did not set the project-local Hugging Face cache before loading; fixed by setting `HF_HOME` and offline mode before imports. Keep using the project cache; do not redirect to the nearly full C: cache.
- Phase 2 approximate-90 ACI gamma 0.001 is the minimum of its calibration grid; the all-valid-step version uses that same selected gamma and has wider bounds. Do not describe it as independently gamma-tuned.
- Phase 3 p(fork) test AUPRC (0.430) is below volatility (0.445), activating the kill/pivot signal. It also overpredicts synthetic odds (mean 0.609 vs 0.442; odds-bin MAE 0.310). The target is model miss, so do not claim calibrated latent fork probability.

## Latest results

- One CPU inference completed for Chronos-Bolt Small (`amazon/chronos-bolt-small`), PEMS-BAY sensor 400001, 14h context and 1h horizon, using a point at the 70% file index. First median forecast: 67.66 mph; observed first step: 68.10 mph. Sanity check only; not a measured benchmark metric.
- Measured timing across 10 repeats: median 0.3125 process CPU seconds and 0.0857 wall seconds per forecast after warm-up.
- Figure: `outputs/phase0/chronos_bolt_pems_bay.png`.
- Phase 1 headline: approximate nominal 90% transition coverage at 60 minutes = 49.7% (n=782); native P10-P90 80% transition coverage = 43.9% (n=782). Published nominal-90 reference is 54.9%, a -5.2 percentage-point descriptive gap; interval and sampling details differ.
- Full results: `outputs/phase1/coverage_by_regime.csv`; plot: `outputs/phase1/coverage_by_regime.png`; report: `outputs/phase1/report.md`; run record: `EXPERIMENTS.md`.
- Phase 2 60-minute transition coverage (n=782): native approximate90 49.7%; ACI approximate90 50.4%; history sampler 73.9%; label-free BMA+ACI paper-style weight 54.9%; label-free BMA+ACI global weight 72.4%; native80 43.9%; ACI native80 44.9%.
- Phase 2 calm coverage (n=3,718): native approximate90 87.6%; ACI approximate90 88.3%; history 81.4%; BMA+ACI paper-style 84.7%; BMA+ACI global 89.6%.
- Detailed Phase 2 outputs: `outputs/phase2/metrics_by_regime.csv`, `headline.csv`, parameter sweep CSVs, `report.md`, `coverage_by_regime.png`, and `width_vs_coverage.png`. All 15 tests pass. History result is 7.1-8.1 points under [1]'s 81-82%; global label-free BMA adaptation is 1.2 points under 73.6% but is not exact/comparable BMA; approximate90 Chronos is 5.2 points under [1]'s 54.9%.
- Phase 2 audit: history 73.9% 60m transition coverage, sample point MAE uses per-horizon sample medians; training neighbors cannot overlap later blocks. The all-valid-step ACI row reaches 58.2% transition coverage, 91.7% calm coverage, 18.39 mph mean width, 48.65 interval score and 4.45 MAE (n=782 transition, 3,718 calm). Phase 2 tests now total 16 and pass.
- Phase 3 test AUPRC: p(fork) 0.4299; volatility 0.4450; recent-drop 0.4527; CUSUM 0.3023. Calm false-alarm rates: 0.123, 0.874, 0.186, 1.000. Selected k=25/L2=0.01, hard threshold 0.168745, test flag rate 18.67%, 15.96% calibration miss fraction.
- Phase 3 synthetic (600 episodes, 1,200 sensor windows): known mean odds 0.442; mean p 0.609; odds-bin MAE 0.310; Brier 0.239; AP vs realized fork events 0.387. All 25 tests pass. Full detail in `outputs/phase3/report.md`.
