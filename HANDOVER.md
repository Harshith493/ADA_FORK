# HANDOVER

## Project goal (5 lines)

1. Implement FORK as a training-free wrapper around frozen time-series foundation models.
2. Detect likely forks from similar past windows and the model's spread.
3. Build weighted scenario paths with dependence within branches.
4. Keep interval and path coverage with online conformal calibration and separate calm/forked states.
5. Reproduce the published traffic failure, evaluate honestly, and deliver a demo and report.

Project brief: [`docs/onepager_summary.md`](docs/onepager_summary.md).

## Current status

- Phase: 2 - baselines.
- Step: Re-checked references [1] and [7], documented missing method details and runtime estimate.
- Status: in progress; no Phase 2 baseline code or run has started. Waiting on one conflict between exact BMA and label isolation.

## Completed (commit references)

- Project one-pager and initial handover: `0111ceb64dd664f9a79291f74da6267977594e75`.
- Phase 0 notes, forecast script, and plot: `3984bbe` (PEMS-BAY CSV and model cache remain local and untracked).
- Phase 0 approval, measured timing, and Phase 1 decision menu: `b2d23c3`.
- Phase 1 option refinement for the paper's Jan-May data period and scoring labels: `e4bb948`.
- Phase 1 code, config, tests, metrics, plot, and report: `538128e` (forecast cache remains local and ignored).
- Phase 2 source review, assumptions, runtime estimate, and no-reuse record: `733e73c`.

## Exactly what to do next

1. Resolve the BMA conflict described at the end of `DECISIONS.md`: choose the label-free BMA-style adaptation, or explicitly allow the paper's training-only destination-regime gate as a baseline exception.
2. After that choice, implement ACI and the history sampler plus the permitted BMA version; extend label-isolation tests and run the Phase 2 evaluation.
3. Reuse Phase 1's test cache; add a separate calibration cache only. Run a small vectorized kNN timing pilot before the full matching/tuning run.
4. Finish with one consolidated metrics table, two plots, tests, short teaching notes, and three questions; wait for Phase 2 review before Phase 3.

## Decisions

- User decisions: Phase 0 approved; Phase 1 choices 1A, 2A, 3A, 4B, 5A; Phase 2 D1-D7 recorded in [`DECISIONS.md`](DECISIONS.md).
- Pending: choose how to handle [1]'s regime-conditioned BMA given the project rule that methods cannot use regime labels/categories. No other Phase 2 decision is pending.

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
- Phase 2 and later commands: not defined yet; add one command per experiment after approval and record them here and in `EXPERIMENTS.md`.
- Phase 2 estimate: test forecasts are cached; likely 5-30 minutes for a 50-origin calibration pass, vectorized neighbor matching, tuning, and metrics. The large work is approximately 4,500 neighbor queries over roughly 30,433 training windows of length 168. Measure a small pilot before launching the full match; if projected runtime exceeds two hours, provide Colab/Kaggle instructions first.

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
- Phase 2 implementation files and one-command runner have not been created; wait for the BMA-method choice first.
- `data/raw/` - downloaded raw public data (not committed).
- `outputs/phase0/` - Phase 0 forecast figure.
- `.cache/huggingface/` - local model cache (not committed).

## Proposed repository structure (not implemented yet)

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
- Synthetic generator: not implemented.
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
- Reference [1] BMA explicitly uses training-estimated probabilities for destination regimes; no official code was found or reused. Exact BMA conflicts with score-only labels unless user authorizes the baseline exception.

## Latest results

- One CPU inference completed for Chronos-Bolt Small (`amazon/chronos-bolt-small`), PEMS-BAY sensor 400001, 14h context and 1h horizon, using a point at the 70% file index. First median forecast: 67.66 mph; observed first step: 68.10 mph. Sanity check only; not a measured benchmark metric.
- Measured timing across 10 repeats: median 0.3125 process CPU seconds and 0.0857 wall seconds per forecast after warm-up.
- Figure: `outputs/phase0/chronos_bolt_pems_bay.png`.
- Phase 1 headline: approximate nominal 90% transition coverage at 60 minutes = 49.7% (n=782); native P10-P90 80% transition coverage = 43.9% (n=782). Published nominal-90 reference is 54.9%, a -5.2 percentage-point descriptive gap; interval and sampling details differ.
- Full results: `outputs/phase1/coverage_by_regime.csv`; plot: `outputs/phase1/coverage_by_regime.png`; report: `outputs/phase1/report.md`; run record: `EXPERIMENTS.md`.
