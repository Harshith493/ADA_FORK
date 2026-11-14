# Design decisions

No research/design decisions have been made by the agent. The following require the user's choice before Phase 1:

1. Transition scoring label: (A) follow the referenced study's horizon-wise thresholds (free-flow above 55 mph, congested below 25 mph; other/mixed outcomes transition), (B) use the one-pager's regime labels exactly once clarified, or (C) define a different scoring-only operational rule. Recommendation: A for direct reproduction, subject to approval.
2. Reproduction sample: (A) mirror the paper's three seeded groups of 30 sensors selected from the top 50 by training congestion frequency and 50 test windows at horizons 3/6/12, (B) evaluate all qualifying chronological test windows with the same sensor rule, or (C) choose another sample size. Recommendation: A first for comparability, then B as a robustness run if time permits.
3. Post-cutoff claim: (A) treat PEMS-BAY/METR-LA strictly as historical replications and defer headline claims to user-approved post-cutoff data, (B) relax the post-cutoff-only requirement, or (C) pause the headline evaluation pending a different dataset. Recommendation: A; the cited model card does not establish no overlap with these traffic datasets.
4. Python setup: (A) create a project `.venv` using Python 3.12 and CPU-compatible PyTorch/Chronos dependencies, (B) continue using the host Python used for Phase 0. Recommendation: A for reproducibility; Phase 0 currently used the host Python.

The Phase 0 example plot's one sensor, date, and index were chosen only to exercise inference; they are not evaluation design decisions.
