# Design decisions

No research/design decisions have been made by the agent. The following require the user's choice before Phase 1:

1. Transition scoring label: (A) follow the referenced study's horizon-wise thresholds (free-flow above 55 mph, congested below 25 mph; other/mixed outcomes transition), (B) use the one-pager's regime labels exactly once clarified, or (C) define a different scoring-only operational rule. Recommendation: A for direct reproduction, subject to approval.
2. Reproduction sample: (A) mirror the paper's three seeded groups of 30 sensors selected from the top 50 by training congestion frequency and 50 test windows at horizons 3/6/12, (B) evaluate all qualifying chronological test windows with the same sensor rule, or (C) choose another sample size. Recommendation: A first for comparability, then B as a robustness run if time permits.
3. Post-cutoff claim: (A) treat PEMS-BAY/METR-LA strictly as historical replications and defer headline claims to user-approved post-cutoff data, (B) relax the post-cutoff-only requirement, or (C) pause the headline evaluation pending a different dataset. Recommendation: A; the cited model card does not establish no overlap with these traffic datasets.
4. Python setup: (A) create a project `.venv` using Python 3.12 and CPU-compatible PyTorch/Chronos dependencies, (B) continue using the host Python used for Phase 0. Recommendation: A for reproducibility; Phase 0 currently used the host Python.

The Phase 0 example plot's one sensor, date, and index were chosen only to exercise inference; they are not evaluation design decisions. Phase 0 was reviewed and approved by the user with no additional notes.

## Phase 1 choices awaiting the user

1. **Transition scoring definition**
   - A. Use the cited paper's traffic rule for each evaluated target period: free-flow when all target observations exceed 55 mph; congested when all are below 25 mph; every other target is transition. Labels remain scoring-only. **Recommend for paper reproduction.**
   - B. Score the endpoint at the forecast horizon only: transition if that future speed is 25-55 mph, stable otherwise. Simpler to explain but changes the paper's target-period definition.
   - C. Use a different scoring-only rule supplied by the user. Allows a domain-specific definition, but is less directly comparable.
2. **Sensor and window sampling**
   - A. Follow the paper: restrict to its stated Jan-May 2017 PEMS-BAY period; rank top 50 sensors by train-only fraction below 25 mph; use three seeded subsets of 30 and 50 held-out windows per subset. **Recommend for close reproduction.**
   - B. Use all available Zenodo dates (through June 30, 2017), all eligible sensors, and all test windows. More data, but differs from the paper's stated period and sample.
   - C. Use Jan-May with the paper's seeded sensor subsets, but score all eligible test windows. A compromise that matches period/sensors while increasing test sample size.
3. **Context length and forecast horizons**
   - A. Use 14 hours of context and 3, 6, and 12 steps (15, 30, 60 minutes), predicting 12 steps once per window. **Recommend for direct reproduction.**
   - B. Use 14 hours and only 12 steps (one-hour headline). Faster and matches the one-pager headline, but omits two published horizons.
   - C. Use a different context length selected by the user and 12 steps only. Tests context sensitivity but adds a design variable and compute.
4. **How to build a nominal 90% interval from Chronos-Bolt**
   - A. Use P10-P90 and call it an 80% interval; report the nominal-90 reproduction as unavailable natively. Cleanest model-native report, but cannot directly compare the published 54.9% number.
   - B. Extrapolate P05/P95 from the outer deciles (e.g. linear tails from P10/P20/P30 and P70/P80/P90), then freeze that rule. Gives an approximate 90% interval, but relies on an unverified tail assumption. **Recommend only if nominal-90 reproduction is required, clearly labeled approximate.**
   - C. Start with P10-P90 and widen it using a calibration set before test. This yields an ACI-style corrected interval, not a native-model interval; report it as the ACI baseline, not as native Chronos.
5. **Chronological train/calibration/test split**
   - A. Use 70/10/20 chronological blocks as the paper states; use train for sensor ranking/history, validation for any interval construction, and test for final scoring. **Recommend for reproducibility.**
   - B. Use 70/15/15 chronological blocks for more calibration data, changing the paper protocol.
   - C. Keep 70/10/20 but split validation chronologically into calibration and tuning sections. More leakage-resistant for later conformal phases, with less data per role.
