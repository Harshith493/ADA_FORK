# HANDOVER

## Project goal (5 lines)

1. Implement FORK as a training-free wrapper around frozen time-series foundation models.
2. Detect likely forks from similar past windows and the model's spread.
3. Build weighted scenario paths with dependence within branches.
4. Keep interval and path coverage with online conformal calibration and separate calm/forked states.
5. Reproduce the published traffic failure, evaluate honestly, and deliver a demo and report.

Project brief: [`docs/onepager_summary.md`](docs/onepager_summary.md).

## Current status

- Phase: 7 - clip fix, demo, and results sheet.
- Step: clipping code, demo cache/app, smoke test, report draft, and runbook draft are present. The clipped Phase 6 replay stopped before writing new results; the committed Phase 6 metrics still describe the unclipped run.
- Status: in progress; user requested a stop at ~4% remaining usage. Do not present old Phase 6 files as clipped results.

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
- Phase 4 branch implementation, config, tests, and research notes: `c79ed1a`.
- Phase 5 conformal implementation, full measured outputs, tests, and research notes: `4948e96`.
- Phase 5 headline table extended with regime-specific widths/scores: `64c33d8`.
- Phase 5 implementation, configuration, tests, outputs, and research notes: see separate implementation commit after the run.
- Phase 5 audit corrections and Phase 6 code, tests, results, and research notes: `9e01e47`.
- Phase 7 implementation/code and report/runbook drafts: `3f697b7` (clipped replay results and final report remain pending).

## Exactly what to do next

1. Compile and run the clipped Phase 6 evaluation: `python scripts\run_phase6.py`. It clips calibration tuning intervals and test intervals to each sensor's training min/max. The interrupted attempt did not update outputs.
2. Regenerate bootstrap/headline files from the completed clipped cache: `python scripts\run_phase6_bootstrap.py` then `python scripts\write_phase6_headline.py`. Recompute summaries with `python scripts\summarize_phase6.py` and `python scripts\summarize_phase6_reliability.py` if their caches are absent.
3. Replace the pending “before clipping” and Phase 6 placeholder values in `docs/results_for_report.md`; update `outputs/phase6/learnings_table.md` and record the final A1 five-line before/after comparison.
4. Finish point-MAE subset labels in generated Phase 2/5/6 tables. Current source changes label Phase 2/5 values, but the existing Phase 5 output CSV/MD were not regenerated; confirm Phase 6 labels say all test windows at 60m.
5. Run `python scripts\build_phase7_demo_cache.py` if `data/cache/phase7_demo_cache.npz` is absent. The cache was built locally already. Run `python -m unittest tests.test_demo -v`; it passed. Launch `streamlit run app\demo.py` and verify Replay, Coverage level, Results, and Limitations tabs in the browser; only import/cache smoke testing has been done.
6. Run `python -m unittest discover -s tests -v` after the output and UI changes. Do not start Phase 8; wait for user review.

## Decisions

- User decisions: Phase 0 approved; Phase 1 choices 1A, 2A, 3A, 4B, 5A; Phase 2 D1-D7 recorded in [`DECISIONS.md`](DECISIONS.md).
- Phases 2-5 are approved. Phase 6 implementation/evaluation is complete with scale-up and METR-LA transfer skipped. User decisions and autonomous choices are in [`DECISIONS.md`](DECISIONS.md).

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
- Phase 6 A-D command: `python scripts\run_phase6.py`; fixed configuration `configs/phase6.json`; outputs under `outputs/phase6/`. It reuses cached forecasts and branches; no Chronos inference is run.
- Phase 6 full test command: `python -m unittest discover -s tests -v` (41 passed). Output summaries: `python scripts\write_phase6_headline.py`, `python scripts\run_phase6_bootstrap.py`, `python scripts\summarize_phase6.py`, `python scripts\summarize_phase6_reliability.py`.
- Phase 6 scale estimate: `python scripts\run_phase6_scaled.py --estimate-only`; estimated 69.5 minutes before new caches, so it was not run. This script is only an estimate/guard, not an all-origin cache builder. Full scaling needs cache generation before the evaluator can consume all origins.
- METR-LA: `python scripts\download_metr_la.py` attempted Zenodo but failed DNS. Manual: download `METR-LA.csv` from https://zenodo.org/records/5146275 to `data/raw/METR-LA.csv`; verify `(Get-FileHash -Algorithm MD5 .\data\raw\METR-LA.csv).Hash.ToLower()` equals `d3adaa79856bf610f25558fc242c7190`. Transfer pipeline remains unavailable because caches/loaders are PEMS-shaped.

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
- `scripts/run_phase4.py`, `configs/phase4_pems_bay.json` - lean Phase 4 branch experiment and fixed settings.
- `src/fork/phase4.py` - label-free branch weights, residual copula paths, intervals, and path scores.
- `tests/test_phase4.py` - branch weight/collapse, copula, deterministic output, label boundary, and no-self-match checks.
- `configs/phase5_pems_bay.json`, `scripts/run_phase5.py` - fixed Phase 5 settings and one-command evaluation.
- `src/fork/phase5.py` - group-specific conformal score, finite-sample level, ACI state, and outcome maturity queue.
- `tests/test_phase5.py` - score cap, state independence, maturity, random-gate coverage, and label-free core checks.
- `outputs/phase5/` - test metrics, gamma sweep, group/path coverage, alpha trace, D8 verdict, and figures.
- `configs/phase6.json`, `scripts/run_phase6.py` - A-D matched alpha sweep, reliability, and bootstrap analysis using frozen caches.
- `src/fork/phase6.py`, `tests/test_phase6.py` - deterministic date-block bootstrap, interpolation, branch reliability and label-free helper tests.
- `outputs/phase6/` - corrected path audit, matched curves/tables, bootstrap CIs, branch calibration, learning table, and plots.
- `scripts/run_phase6_scaled.py`, `configs/phase6_scaled.json` - scale-up runtime estimate; all-origin cache generation is not implemented here.
- `scripts/download_metr_la.py` - public Zenodo downloader with MD5 verification; network attempt failed in this environment.
- `app/demo.py` - Streamlit cached-output demo; app code exists but tabs have not been visually verified.
- `scripts/build_phase7_demo_cache.py`, `data/cache/phase7_demo_cache.npz` - local replay cache built from existing PEMS-BAY forecasts and branch cache; no inference/download.
- `docs/results_for_report.md`, `docs/RUNBOOK.md` - drafted; results sheet still contains pre-clipping references/placeholders until the rerun completes.
- `data/cache/phase5_branch_outputs.npz` - ignored calibration branches and weights for the Phase 5 replay; Phase 4 supplies the test branch cache.
- `outputs/phase4/` - Phase 4 test metrics, comparison, and example plot.
- `data/cache/phase4_branch_cache.npz` - ignored per-window branch paths/weights, generated by the Phase 4 run.
- `tests/test_phase3.py` - no-label dependency, no-self-match, train-only, toy k-means, PR ties, and determinism tests.
- `outputs/phase3/` - test metrics, calibration sweep, PR curve/table, example neighbor branches, synthetic odds comparison, and report.
- `data/cache/phase3_train_5000_forecasts.npz`, `phase3_feature_cache.npz`, `phase3_fork_outputs.npz` - ignored local caches; final cache has cal/test p(fork) and minority shares for Phase 4/5.
- `data/cache/phase2_consecutive_aci_sampled_bounds.npz` - sampled-origin bounds after all-valid-step ACI updates; ignored local cache.
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
- METR-LA download was attempted in Phase 6 but DNS resolution failed. Manual action if transfer is desired: open https://zenodo.org/records/5146275, download `METR-LA.csv`, place it at `data/raw/METR-LA.csv`, then verify `(Get-FileHash -Algorithm MD5 .\data\raw\METR-LA.csv).Hash.ToLower()` equals `d3adaa79856bf610f25558fc242c7190`. Transfer remains skipped because data/cache shapes are PEMS-specific.
- Synthetic two-regime generator: implemented and scored in Phase 3 (not a separate data file).
- Phase 3 synthetic two-regime/fork generator is implemented in `src/fork/phase3.py` and evaluated in `outputs/phase3/synthetic_metrics.csv`; standalone reusable data cache is not needed.
- No manual account/token is needed for the above public files. Caltrans PeMS 2026 requires the user's manual account/download step and remains stretch-only.
- Data citation/license metadata on the Zenodo record is incomplete; do not redistribute the raw data without checking source terms.
- Chronos model training overlap with PEMS-BAY/METR-LA is not resolved by the model card. Treat the historic datasets as replication only, not post-cutoff headline evidence.

## Known issues / failed attempts / do not retry

- Phase 4 run results are committed under `outputs/phase4/`; the predeclared D8 coverage match was not met, though its mean width was smaller.
- Phase 4 uses an explicit ±5 percentage-point definition of “similar coverage” for D8 and reuses the calibration-selected threshold for branch collapse and interval inclusion; see `DECISIONS.md`.
- Phase 5 whole-path coverage is far below 90% (about 66% in main-gate groups); score support caps and low effective path sample size may be limiting factors. Do not claim 90% path coverage.

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
- Phase 6 scale-up is estimate-only: current sample workload projects 69.5 minutes for 43,440 all-origin windows at 30-minute stride, before creating forecast/branch caches. The `run_phase6_scaled.py` script is only a runtime preflight, not a complete cache-generation runner. Do not describe the scale-up as implemented or measured.
- Phase 6 METR-LA download failed at DNS resolution (`getaddrinfo failed`); do not retry network download in this restricted runtime. If downloaded manually, transfer requires refactoring the PEMS-shaped cache/loaders and frozen Phase 3/4/5 pipeline.

## Latest results

- Phase 4 test result: 82.1% transition coverage, 90.7% calm coverage, 16.83 mph mean width; outside the documented ±5 point coverage match. Outputs in `outputs/phase4/`.
- Phase 5 main p(fork) gamma=0.01. Test group coverage: 88.4% calm-flagged / 89.3% forked-flagged; regime coverage 89.8% free-flow / 82.2% transition / 92.0% congested. Whole-path coverage is only 66.3% / 66.4% in the two groups.
- Random gate group interval coverage: 89.4% / 88.7%, but whole-path coverage 66.6% / 64.7%. FORK weighted-median MAE 3.000 mph vs Chronos q50 3.098 mph.
- D8(e): transition coverage misses 85%; no Chronos ACI baseline is within ±2 points of FORK transition coverage, so no matched-coverage sharpness claim. Phase 3 kill signal remains active (p(fork) AUPRC .4299 < volatility .4450 and recent-drop .4527).
- Full Phase 5 outputs: `outputs/phase5/headline_60m.md`, `results.md`, `whole_path_coverage.csv`, `selected_gamma.csv`, `D8e_verdict.md`, and plots. The 35-test unit suite passed; full run used cached Chronos forecasts and generated calibration branch paths in about 0.3 minutes.

## Phase 6 latest result and commands

- Phase 6 status: A-D measured on 4,500 fixed sampled test windows; no Chronos inference recomputed. Corrected 60m table: FORK transition coverage 82.2% [79.0,86.3], group coverage 88.4% [85.6,91.0] calm / 89.3% [86.2,92.8] forked, transition width 34.33 [30.78,36.77] mph, interval score 58.10 [49.39,67.41], MAE 3.89 [2.83,5.09]. Scaled Chronos gated: 74.3% [67.0,82.9] transition, 90.1% [87.1,93.2] calm / 89.2% [82.2,96.7] forked, width 90.98 [75.34,109.47], interval score 159.85 [128.32,187.83], q50 MAE 4.45 [3.10,5.88]. CI method: 1000 day-block bootstrap resamples, seed 6060.
- 60m matched transition at 70%: FORK width 27.54 vs Chronos 74.64 mph (63.1% lower); 80% has no range overlap (Chronos max 78.4%). At matched calm 90%, FORK 60m width is 12.87 vs 16.14 mph; 15m is wider and 30m nearly equal.
- Minority Brier 0.2141; for predicted weights in [.25,.35], observed nearest-minority rate 33.1% (n=1,157). Five fixed-width bins; first bin empty due branch threshold.
- A2: max-score path-region vs joint-12 distinction; FORK's region coverage 66% because capped branch nonconformity cannot extend beyond empirical branch support. Scaled Chronos path region reaches ~89.5% by gate; joint-12 remains 63.1% calm / 75.7% fork.
- Tests: `python -m unittest discover -s tests -v` (41 passed, 7.8 sec). Main experiment: `python scripts/run_phase6.py` (~7.2 minutes). Postprocess/manual commands: `python scripts/run_phase6_bootstrap.py`, `python scripts/write_phase6_headline.py`, `python scripts/summarize_phase6.py`, `python scripts/summarize_phase6_reliability.py`.
- Scale-up command: `python scripts/run_phase6_scaled.py --estimate-only` (69.5 min estimate; not run). METR-LA downloader: `python scripts/download_metr_la.py` (failed due DNS; manual steps above).
- Full tables: `outputs/phase6/headline_60m_ci.md`, `day_block_bootstrap_ci.csv`, `matched_coverage.csv`, `matched_calm_coverage.csv`, `branch_reliability.csv`, `learnings_table.md`. The p(fork) AUPRC kill signal stays active; 85% transition and 90% FORK path-region targets were missed.

## Phase 7 stop-point status (2026-11-15)

### Done

- Added `clip_interval` and wired per-sensor training-block min/max clipping into Phase 6 calibration objective and test width/score/coverage calculations. No calibration/test extrema enter the physical limits.
- Added the clipping unit test and Phase 7 code for replay, coverage, results, and limitations tabs. Demo preset label use is display-only and disclosed in the UI.
- Built `data/cache/phase7_demo_cache.npz` locally from cached forecasts/branches plus PEMS-BAY contexts; no inference or download. `python -m unittest tests.test_demo -v` passed and imported the app functions/loaded one 168-step context plus 12-step future.
- Drafted `docs/results_for_report.md` and `docs/RUNBOOK.md`; added the Phase 7 A3 whole-path limitation to `DECISIONS.md`.

### In progress / not done

- Clipped Phase 6 rerun and A1 before/after comparison: **not complete**. The latest attempt is stopped; no live Python process remains. Phase 6 output CSV/MD files still contain the pre-clip results.
- Results and Limitations Streamlit tabs: code is present, but the running app/browser tabs have not been verified. Smoke test only checked import/cache load.
- Report and runbook: files exist, but report contains pending clipped results and before/after reference; finalize after the rerun.
- A3 DECISIONS entry: done. A2 MAE subset labels: source/report edits are present, but generated outputs have not all been regenerated or audited. Label Phase 5 MAE “all test windows at 15/30/60m” and Phase 6 60m MAE “all test windows at 60m,” not transition-only.
- No estimate of remaining Codex usage is available. User directed stop at about 4%; do not start another long task in this handover step.

### Exact remaining commands

1. `python -m py_compile scripts\run_phase6.py app\demo.py scripts\build_phase7_demo_cache.py`
2. `python scripts\run_phase6.py` (longer than the user's requested short stop step; run only when authorized/available; writes clipped A-D metrics from caches, no Chronos inference).
3. `python scripts\run_phase6_bootstrap.py` then `python scripts\write_phase6_headline.py`; `python scripts\summarize_phase6.py`; `python scripts\summarize_phase6_reliability.py`.
4. Update `docs/results_for_report.md`, `outputs/phase6/learnings_table.md`, Phase 5/6 point-MAE subset labels; add the before/after A1 five-line note.
5. `python scripts\build_phase7_demo_cache.py` only if cache missing; `python -m unittest tests.test_demo -v`; then `streamlit run app\demo.py` and verify all tabs.
6. `python -m unittest discover -s tests -v`.

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
