# Credits and reused components

## Phase 0

- **Chronos-Bolt Small model weights** (`amazon/chronos-bolt-small`): used only for the requested inference sanity check. Official Hugging Face model card states Apache-2.0. Citation: Ansari et al., *Chronos: Learning the Language of Time Series* (2024), linked from the model card. The official repository is `amazon-science/chronos-forecasting` (Apache-2.0).
- **Chronos Python package** (`chronos-forecasting` 2.3.2): inference dependency, not baseline implementation. Official Amazon repository is Apache-2.0 licensed.
- **PEMS-BAY CSV**: downloaded from Zenodo record 5146275, `PEMS-BAY and METR-LA in csv`. Zenodo's visible license metadata is incomplete; raw data is not redistributed in Git.

## Baseline code check

- Reference [1] is accessible on arXiv. No official companion code repository or code license was located in the paper record/page during Phase 0. No reference or baseline code has been reused. If code is identified later, record its exact URL and license, then wait for user approval before reusing it.

## Phase 2 source re-check

- Re-read reference [1] from its arXiv HTML/PDF and searched the paper title for an author-linked repository. No official code was found or reused.
- Rechecked reference [7] from the official NeurIPS paper page/PDF for the ACI update rule. No external ACI implementation was reused; Phase 2 will implement the rule from the paper description if the method conflict is resolved.
