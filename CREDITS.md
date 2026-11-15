# Credits and reused components

## Phase 0

- **Chronos-Bolt Small model weights** (`amazon/chronos-bolt-small`): used only for the requested inference sanity check. Official Hugging Face model card states Apache-2.0. Citation: Ansari et al., *Chronos: Learning the Language of Time Series* (2024), linked from the model card. The official repository is `amazon-science/chronos-forecasting` (Apache-2.0).
- **Chronos Python package** (`chronos-forecasting` 2.3.2): inference dependency, not baseline implementation. Official Amazon repository is Apache-2.0 licensed.
- **PEMS-BAY CSV**: downloaded from Zenodo record 5146275, `PEMS-BAY and METR-LA in csv`. Zenodo's visible license metadata is incomplete; raw data is not redistributed in Git.

## Baseline code check

- Reference [1] is accessible on arXiv. No official companion code repository or code license was located in the paper record/page during Phase 0. No reference or baseline code has been reused. If code is identified later, record its exact URL and license, then wait for user approval before reusing it.

## Phase 2 source re-check

- Re-read reference [1] from its arXiv HTML/PDF and searched the paper title for an author-linked repository. No official code was found or reused.
- Rechecked reference [7] from the official NeurIPS paper page/PDF for the ACI update rule. No external ACI implementation was reused; the update rule is implemented locally from the paper description.
- Phase 2 baselines and matching are original local implementations. No third-party baseline code, copula code, or other reference implementation was copied or imported.
- Sources: [1] https://arxiv.org/html/2606.18367; [7] https://papers.neurips.cc/paper_files/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf. Chronos runtime and weights are the same already credited above.

## Phase 3

- The calendar embedding, train-only matching extension, deterministic two-means, logistic regression, volatility/recent-drop/CUSUM scores, synthetic generator, and metrics are implemented locally with NumPy/pandas. No additional package or external code was used.
- Chronos-Bolt weights/inference and the PEMS-BAY data are previously credited above. Phase 3 reused the Phase 1 test-forecast cache and Phase 2 calibration-forecast cache.

## Phase 4

- No external source code or new package was reused. Branch construction, residual-correlation shrinkage, Gaussian-copula sampling, energy score, and variogram score are local implementations. The local matching, k-means, quantile sampling, and interval-score components are reused from earlier project phases.

## Phase 5

- No external code or package is reused. The online ACI update, finite-sample score threshold, maturity queue, branch-union score, and evaluation loop are implemented in this project. Chronos forecast caches and the project's Phase 3/4 outputs are reused.

## Phase 6

- No external code or package was reused. The unbounded scaled Chronos score, matched-coverage interpolation, day-block bootstrap, and branch-weight reliability analysis are local implementations. Existing Phase 1/3/4/5 caches and Phase 5 replay functions are reused.
- METR-LA raw data was not downloaded or redistributed. The network attempt failed at DNS resolution; a downloader script uses the same public Zenodo record as PEMS-BAY and checks the previously verified MD5.
