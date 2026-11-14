# FORK one-pager summary

## Goal

Build a training-free wrapper around a frozen time-series foundation model (MVP: Chronos-Bolt) for forecasts near regime changes. FORK should detect likely forks from similar past windows and the TSFM's spread, represent them as weighted future branches with within-branch dependence, then use online conformal calibration with separate calm/forked states.

## Evidence and target

The cited 2026 study reports that Chronos-Bolt on PEMS-BAY has 54.9% empirical coverage for nominal 90% intervals during transitions at a 12-step (one-hour) horizon. It reports 73.6% for BMA + ACI and 81-82% for a historical conditional sampler. These are published reference values, not project results. The goal is to reproduce the failure before evaluating FORK and baselines.

## MVP scope

- PEMS-BAY, METR-LA, and a synthetic regime-switching generator.
- Chronos-Bolt only; CPU-only Windows workstation, Intel i7-13700H, 14 cores, 16 GB RAM.
- Regime/transition labels are for scoring only; no label can enter FORK features or state.
- Strict chronological splits; calibration before test; online updates use past observations only.
- Evaluate coverage, interval score, Brier, energy and variogram scores, point accuracy, sharpness, and the one-pager's oracle and kill/pivot signals.
- Work in phases 0-7; show results, plots, what to understand, and three review questions at each phase end; wait for approval before advancing.

## Research and reuse constraints

Verify cited papers, models, datasets, access, licenses, and overlap before use. Check for official baseline code and its license; ask before reusing it. Implement FORK from scratch. Record design choices in `DECISIONS.md`, runs in `EXPERIMENTS.md`, reused material in `CREDITS.md`, and continuation state in root `HANDOVER.md`.

## Known data-period issue

PEMS-BAY and METR-LA are historical (2017 and 2012 in the cited study), while Chronos is pretrained on a large public-data corpus. Chronos' model card reports benchmark datasets were held out from training but does not establish whether these traffic datasets were excluded. These historical datasets can support replication, but cannot by themselves establish post-training-cutoff performance. A newer dataset is needed for that headline claim; Caltrans PeMS 2026 requires the user's manual account/download step.
