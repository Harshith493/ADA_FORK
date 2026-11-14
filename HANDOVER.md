# HANDOVER

## Project goal (5 lines)

1. Implement FORK as a training-free wrapper around frozen time-series foundation models.
2. Detect likely forks from similar past windows and the model's spread.
3. Build weighted scenario paths with dependence within branches.
4. Keep interval and path coverage with online conformal calibration and separate calm/forked states.
5. Reproduce the published traffic failure, evaluate honestly, and deliver a demo and report.

Project brief: [`docs/onepager_summary.md`](docs/onepager_summary.md).

## Current status

- Phase: 0 - setup and data access.
- Step: verify public sources, establish a runnable CPU forecast, and propose the repo/environment/data plan.
- Status: done; Phase 1 is waiting on the user's design choices below.

## Completed (commit references)

- Project one-pager and initial handover: `0111ceb64dd664f9a79291f74da6267977594e75`.
- Phase 0 notes, forecast script, and plot: `3984bbe` (PEMS-BAY CSV and model cache remain local and untracked).

## Exactly what to do next

1. Wait for the user's choices on transition labels, replication sampling, and the post-cutoff headline constraint.
2. Record those choices and rationale in `DECISIONS.md`.
3. Reconcile the PEMS-BAY CSV's observed date span (through 2017-06-30) against the paper's stated Jan-May period before fixing the reproduction subset.
4. At the start of Phase 1, build a reproducible chronological dataset/split and baseline forecast pipeline; do not add FORK logic yet.
5. After Phase 1, save metrics and plots, update this file, explain the findings, ask three learning questions, and wait for approval.

## Decisions

- User decisions: none yet; see [`DECISIONS.md`](DECISIONS.md).
- Pending: transition definition; reproduction split/window/sensor sampling; how to handle the post-cutoff-only headline requirement; proposed Python environment; approval to proceed to Phase 1.

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
- Phase 1 and later commands: not defined yet; create one command per experiment after approval and record each here and in `EXPERIMENTS.md`.

## File map

- `AGENTS.md` - short project rules for future agents.
- `HANDOVER.md` - current progress and exact continuation steps.
- `Aporia_One-Pager_FORK.pdf` - original project brief.
- `docs/onepager_summary.md` - compact brief and confirmed scope.
- `DECISIONS.md` - user-approved design choices and open questions.
- `EXPERIMENTS.md` - reproducible experiment configurations and outcomes.
- `CREDITS.md` - external components, sources, and licenses.
- `scripts/phase0_forecast.py` - one CPU-only Chronos-Bolt sanity forecast.
- `data/raw/` - downloaded raw public data (not committed).
- `outputs/phase0/` - Phase 0 forecast figure.
- `.cache/huggingface/` - local model cache (not committed).

## Data status

- PEMS-BAY: present at `data/raw/PEMS-BAY.csv` (85,718,439 bytes, Zenodo record 5146275). The CSV covers 2017-01-01 through 2017-06-30.
- METR-LA: not downloaded; public at the same Zenodo record (72.8 MB) and linked through DCRNN.
- Synthetic generator: not implemented.
- No manual account/token is needed for the above public files. Caltrans PeMS 2026 requires the user's manual account/download step and remains stretch-only.
- Data citation/license metadata on the Zenodo record is incomplete; do not redistribute the raw data without checking source terms.
- Chronos model training overlap with PEMS-BAY/METR-LA is not resolved by the model card. Treat the historic datasets as replication only, not post-cutoff headline evidence.

## Known issues / failed attempts / do not retry

- First model load tried the C: Hugging Face cache and failed for lack of disk space. It is fixed by setting `HF_HOME` to the project-local cache; do not restore the default cache on this machine.
- First script version used the wrong `predict_quantiles` argument name; fixed to `inputs=`.
- The model's native quantile grid spans 0.1-0.9. Phase 0 plot shows its 80% P10-P90 band; it is not evidence for 90% interval coverage.
- The package and model were obtained from the official Amazon Chronos distribution; no code from reference [1] or another baseline was reused.
- No tests have been run; this was a one-off inference sanity check.

## Latest results

- One CPU inference completed for Chronos-Bolt Small (`amazon/chronos-bolt-small`), PEMS-BAY sensor 400001, 14h context and 1h horizon, using a point at the 70% chronological index. First median forecast: 68.92 mph; observed first step: 68.10 mph. Sanity check only; not a measured benchmark metric.
- Figure: `outputs/phase0/chronos_bolt_pems_bay.png`.
- Full experiment results do not exist yet; future results belong in `EXPERIMENTS.md` and a report.
