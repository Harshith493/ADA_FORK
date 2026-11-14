# Design decisions

The user approved the Phase 1 decisions recorded below. The earlier options are retained as the audit trail; selected methods and assumptions are listed in the final section.

1. Transition scoring label: (A) follow the referenced study's horizon-wise thresholds (free-flow above 55 mph, congested below 25 mph; other/mixed outcomes transition), (B) use the one-pager's regime labels exactly once clarified, or (C) define a different scoring-only operational rule. Recommendation: A for direct reproduction, subject to approval.
2. Reproduction sample: (A) mirror the paper's three seeded groups of 30 sensors selected from the top 50 by training congestion frequency and 50 test windows at horizons 3/6/12, (B) evaluate all qualifying chronological test windows with the same sensor rule, or (C) choose another sample size. Recommendation: A first for comparability, then B as a robustness run if time permits.
3. Post-cutoff claim: (A) treat PEMS-BAY/METR-LA strictly as historical replications and defer headline claims to user-approved post-cutoff data, (B) relax the post-cutoff-only requirement, or (C) pause the headline evaluation pending a different dataset. Recommendation: A; the cited model card does not establish no overlap with these traffic datasets.
4. Python setup: (A) create a project `.venv` using Python 3.12 and CPU-compatible PyTorch/Chronos dependencies, (B) continue using the host Python used for Phase 0. Recommendation: A for reproducibility; Phase 0 currently used the host Python.

The Phase 0 example plot's one sensor, date, and index were chosen only to exercise inference; they are not evaluation design decisions. Phase 0 was reviewed and approved by the user with no additional notes.

## Phase 1 options presented to the user

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

## Phase 1 decisions (approved by user)

- Selected options: 1A, 2A, 3A, 4B, 5A.
- Use the paper's transition rule separately for each forecast horizon: free-flow if every observed speed in the target period is above 55 mph; congested if every speed is below 25 mph; otherwise transition. These labels are computed after forecasting and passed only to scoring.
- Restrict PEMS-BAY to timestamps from 2017-01-01 through 2017-05-31 inclusive. This follows the paper's stated period; the source CSV continues through June 30. PEMS-BAY (2017) may overlap Chronos pretraining data, so this phase is historical reproduction, not post-cutoff evaluation.
- Rank the 325 sensors by training-only fraction below 25 mph; for seeds 42, 43, and 44, sample 30 sensors without replacement from the top 50. Sample 50 valid test-window origins once using seed 42 and reuse these origins across all sensor subsets. The paper does not say how its 50 test windows were sampled; uniform random sampling without replacement is the recorded implementation assumption.
- Use 168 context steps (14 hours), forecast 12 steps once, and score 3, 6, and 12-step horizons. Each horizon's regime label uses its observed target period (the first h steps).
- Approximate nominal-90 interval method: request the native q10, q20, q50, q80, q90 values. Linearly extrapolate the quantile function by half of the adjacent decile gap: q05 = q10 - 0.5 * (q20 - q10); q95 = q90 + 0.5 * (q90 - q80). This assumes the quantile function stays locally linear from q10 to q05 and q90 to q95. The result is approximate and has no native 90% or conformal coverage guarantee. Report this only as "approximate nominal 90%".
- Report native q10-q90 as "native 80%" in separate rows; never combine its coverage with approximate nominal-90 coverage.
- Use chronological 70/10/20 train/calibration/test row blocks. Sensor ranking uses train only. Phase 1's frozen quantile extrapolation does not fit on calibration data; that block is reserved for later calibration phases.
- The initial run caches every requested horizon's 5 native quantiles for all selected sensor/window pairs. Results and plot are written under `outputs/phase1/`.

## Phase 2 reference review and implementation notes (2026-11-15)

### Verified against reference [1]

- Historical conditional baseline: per-sensor empirical conditional distribution `P(speed[t+h] | speed[t])`, estimated from training data; draw 100 samples. The paper does not give a continuous-speed matching rule, bins/kernel, a neighbor count, or a recent-window embedding.
- BMA: from training data, estimate per-sensor probabilities `P(speed[t+h] in R | speed[t])` for destination regimes `R`; at test time replace a fraction of Chronos pseudo-samples with draws from the corresponding historical conditional distribution, weighted by transition probability. The paper says `w` is in `[0.1, 0.5]` and selected using 10 held-out windows. It does not state the `w` selection objective, exact conditioning estimator/binning, how the 10 windows or sensor are selected, or a fully reproducible pseudo-sample implementation.
- **Regime information:** yes. The BMA procedure explicitly conditions on future destination regime `R` and uses historical regime-conditional probabilities. It does not use the true test-window regime label at inference; it estimates regime probabilities from training data and current speed. That still conflicts with this project's stricter rule that methods must not use regime labels/categories.
- BMA + ACI: the paper uses global ACI after BMA, not a separate ACI state by regime. The paper does not report a numeric ACI `gamma` or detailed initialization for that experiment. Reference [7] gives the update `alpha[t+1] = alpha[t] + gamma * (alpha_target - miss[t])`, with `alpha[1] = alpha_target` in its experiments.
- Official code: no author-linked or paper-linked companion repository was found on the arXiv record/source or in the targeted search. No external reference code has been reused.

### Phase 2 implementation assumptions to disclose

- History sampler per D4: because [1] only specifies conditioning on current speed and does not provide a matching estimator, use our version of [1]: z-score each sensor's 168-step recent window using train-only sensor mean/std; Euclidean k-nearest-neighbor search over training-block windows; use the neighbors' actual next-hour paths. Planned `k` sweep: `[5, 10, 20, 50]`, choose by mean 90% interval score on calibration only. Implement matching in reusable `src/fork/window_matching.py`.
- ACI `gamma` sweep: `[0.001, 0.005, 0.01, 0.02]`; choose lowest sequential interval score on calibration only. Initialize alpha to nominal miscoverage and apply [7]'s update. Use one state per sensor and method stream, shared across 15/30/60-minute horizons and regimes. Forecasts at one origin all use the same alpha; a miss updates state only once its target timestamp is before a later forecast origin. The width mapping from alpha to a symmetric interval radius is an implementation guess (`radius * sqrt(target_alpha/current_alpha)`) because the references do not specify it for these forecast intervals.
- For BMA weight calibration, planned grid `[0.1, 0.2, 0.3, 0.4, 0.5]`. Since [1] omits its tuning objective and sensor/window selection, use calibration-only interval score; paper-style uses the first sensor in seed-42's selected group and the first 10 calibration windows, while our global version pools all selected sensors/windows.
- The exact paper BMA conflicts with the no-regime-label rule. The user approved a label-free adaptation. Both requested weight-tuning variants use training-window neighbor futures and fixed-weight replacement of Chronos pseudo-samples, without [1]'s destination-regime gate.
- Phase 1's test forecast cache is available and must be reused. It contains 4,500 sensor-window-seed forecasts (3 x 30 x 50) and five quantiles; it contains no calibration forecasts. To tune gamma/k/w without test leakage, add a separate cache of 50 calibration origins x the same sensor groups. This does not recompute Phase 1 test forecasts.
- Runtime estimate before any Phase 2 run: cached test forecasts need no inference. A 50-origin calibration forecast pass is about 150 batches of 30 sensors; Phase 1's analogous 150-batch pass completed in about 18 seconds including model loading. Exact kNN work is roughly 4,500 query windows x 30,433 train windows x 168 features before vectorization; estimate 5-30 minutes total for calibration matching, tuning and metrics with vectorized NumPy, under the 2-hour threshold. Run a small timing pilot before launching that matching pass; if the pilot projects over 2 hours, stop and give Colab/Kaggle steps.

### User resolution and Phase 2 method choices (approved)

- User chose the label-isolated option: implement a label-free BMA-style adaptation and label it exactly **"our label-free adaptation of BMA, not exact BMA from [1]"**.
- Exact difference from [1]: [1] estimates training-only destination-regime probabilities `P(speed[t+h] in R | speed[t])` and uses that probability as a gate for injecting historical samples. Our adaptation does not create or use regime categories or gate by predicted regime. It takes empirical future paths from the training-block kNN match and replaces a fixed weight of Chronos pseudo-samples at every window. The weight is selected on calibration only. No true test label enters any method.
- D3 interpretation: keep one ACI alpha state per sensor and method stream, shared across 15/30/60-minute horizons and all regimes. Outcomes update only after their timestamps have matured before a subsequent origin. ACI streams remain separate for approximate-90 ACI, native-80 ACI, and each BMA+ACI weight variant.
- k, gamma and w candidate grids/objectives in the implementation-notes section above are accepted as the documented implementation guesses. No test data may affect their selection.

### Phase 2 measured selections and explicit guesses

- Selected calibration values: `k=50`; global adaptation weight `w=0.5`; paper-style 10-window adaptation weight `w=0.1`. Gamma selections: approximate-90 ACI `0.001`, native-80 ACI `0.005`, paper-style BMA+ACI `0.001`, global BMA+ACI `0.001`.
- The paper says its BMA gate uses training-only destination-regime labels/categories to estimate `P(speed[t+h] in R | speed[t])`; it does not use the true future test label. FORK's rule excludes even that gate, so our method must be labelled **"our label-free adaptation of BMA, not exact BMA from [1]"**. It uses no regime categories or predicted regime probabilities and replaces a fixed fraction of model pseudo-samples with empirical training-neighbor future paths.
- Guesses: 168-point globally z-scored recent context; Euclidean distance; exact search among timestamp-contiguous training origins; k grid and calibration interval-score objective; paper-style first sensor from seed-42 group and first 10 sampled calibration windows; both BMA weight objectives use mean 60-minute interval score; q pseudo-samples use inverse-CDF linear interpolation and constant outer tails; ACI converts alpha to radius using the symmetric square-root rule above; calibration/test origins are uniform random samples from valid chronological windows (test origins unchanged from Phase 1).
- No official baseline code was reused. Full measured results and sweep tables: `outputs/phase2/report.md`, `outputs/phase2/metrics_by_regime.csv`, and the adjacent `*_sweep.csv` files.

### Phase 2 user audit

- History-sampling candidate futures come from the training block only. Candidate origins are timestamp-contiguous and require their full context and next-hour future to end at or before the training boundary. As calibration and test are later chronological blocks, neighbor windows cannot overlap those blocks; overlaps among training candidate windows are allowed.
- Sample-based point MAE is the MAE of the per-horizon median across neighbor futures. For BMA-style mixtures it is the per-horizon median across 100 mixed paths; Chronos and ACI use q50.
- Approximate-90 ACI selected gamma 0.001, at the minimum of the tested grid. The original Phase 2 test loop updated only at sampled test origins. An added variant forecasts each valid consecutive five-minute test origin and updates after all matured horizon outcomes. It is evaluated on the same sampled origins. Its 60-minute transition coverage is 58.2%, calm coverage 91.7%, mean width 18.39 mph, interval score 48.65, and MAE 4.45. It uses the calibration-selected gamma without a separate all-step gamma tune.
- No prior “Autonomous choices” section existed. Review of Phase 2 configs/code found no autonomous tuning that used test outcomes or transition labels; test outcomes were used only for online ACI updates after maturity and for scoring.

## Autonomous choices

### Phase 5 user decisions and autonomous choices

- User notes: Phase 4 bands are treated as native P10–P90 (80%) for this phase's starting forecasts; Phase 5 calibrates to a 90% target. Freeze Phase 4 settings at alpha=1.0, collapse/branch threshold=0.1, and copula shrinkage=0.5; its 10-origin tuning and near-flat sweeps are limitations. Phase 3 kill signal remains active; p(fork) is a risk score and the primary gate, with the requested ablations.
- Use cached Chronos q forecasts for calibration/test and cached Phase 4 test branch paths. Generate missing calibration branch paths with the same 50 sampled origins/sensors and fixed Phase 4 settings; never rerun Chronos forecasts.
- Gate recent-drop threshold is the calibration quantile that matches the p(fork) calibration flag rate, a label-free choice. Random gate uses exactly that number of flags per block with a fixed-seed permutation. Branch-weight gate means two non-collapsed branches have positive weights. p(fork) uses frozen threshold 0.168745. No-gate is one pooled state.
- Oracle reference gate assigns transition outcomes using true 60-minute regime labels, solely for its explicitly labelled oracle ablation and final scoring. Its gamma is inherited from the main p(fork) method; no oracle-specific tuning is performed.
- Define conformal score as the minimum tau in [0,1] whose central per-branch empirical intervals (after dropping branches below 0.1 weight) cover the observed value. Use bisection to 1e-6 precision. If no branch's full sample support covers the value, cap the score at 1; tau=1 then gives the empirical full-support interval and may still miss, which is counted as an error.
- For whole-path calibration, score is the maximum of the 12 marginal scores and error means at least one horizon lies outside its branch-union interval at the selected common tau. Report split intervals at 15/30/60 minutes.
- Use finite-sample corrected buffer quantile rank `ceil((n+1)*(1-alpha_t))`, clipped to [1,n]. With no matured score history, initialize tau=0.9. Clip ACI alpha state to [0.01,0.99]. Schedule 15/30/60/path outcomes at origin+3/+6/+12/+12 rows; process outcomes with maturity time <= the current origin before prediction.
- Tune gamma separately per non-oracle method over {0.005,0.01,0.02,0.05} by lowest calibration mean 90%-interval score across 15/30/60 minutes. For oracle, inherit the main p(fork) gamma to prevent oracle labels from influencing tuning. Keep buffers and ACI updates pooled by gate group across sensors.
- For calm-collapse ablation, use the branch whose sample median is nearest the Chronos q50 path (the Phase 4 cache did not store the anchor index) when p(fork) says calm; retain both branches when flagged forked.
- Matched-transition-coverage sharpness verdict uses a ±2 percentage-point difference in observed test transition coverage; if outside that band, report no matched-coverage comparison instead of tuning on test. “At least 10% sharper” means at least 10% lower mean 60-minute interval width.
- All gates, score construction, tuning, and interval formation are label-free except the isolated oracle reference. Labels otherwise enter only final regime-stratified scoring.
- Measured Phase 5 gamma choices: p(fork)=0.01; recent-drop=0.005; branch-weight=0.005; random=0.005; no-gate=0.005; p(fork) calm-collapse=0.005; Chronos ACI no-gate=0.005; Chronos ACI p(fork)=0.005. Oracle inherited 0.01.
- Measured outcome: p(fork) interval coverage is 88.4% calm-flagged / 89.3% forked-flagged overall, but true-transition 60m coverage is 82.2% and whole-path group coverage is only 66.3% / 66.4%. Random group interval coverage is near 90% (89.4% / 88.7%), but whole-path coverage remains about 65%. The 85% transition target and 90% whole-path target were missed. The p(fork) AUPRC kill signal remains active (0.4299 vs 0.4450 volatility and 0.4527 recent-drop). Full outputs are under `outputs/phase5/`.

### Phase 4 user-directed pivot and implementation choices

- The Phase 3 kill signal remains active: p(fork) did not outperform volatility or recent-drop in AUPRC. Proceed with Phase 4, report that result, and treat p(fork) only as a risk score. Phase 4 does not use p(fork); Phase 5 will compare gates.
- Keep labels entirely out of branch formation and calibration tuning. The Phase 4 branch API takes only neighbor paths, Chronos quantiles, training normalization, and numeric tuning values. Test labels are read only after paths and intervals have been built, for scoring.
- Use the user's k=25 and seed 4404. Tune alpha in {0.5,1,2} and a shared collapse/minimum-interval branch-weight threshold in {0,.05,.1,.2} on the first 10 sampled calibration origins across the three selected sensor groups. This is a deterministic calibration-only subsample to keep the experiment lean.
- Select alpha and threshold by mean 60-minute approximate-90 interval score on that calibration subset. Select Gaussian-copula shrinkage from {0,.5,1} by calibration energy score after freezing alpha and threshold. The shared threshold controls both branch collapse and interval union branch inclusion.
- The closest cluster center to the Chronos median uses Chronos quantile pseudo-samples; the other uses its neighbor-cluster center plus within-cluster residuals. This is the operational interpretation of “Chronos-anchored” and uses no labels.
- Estimate pooled temporal correlation from the cached Phase 3 train-only Chronos forecast residuals, and branch-local correlations from neighbor paths' within-cluster residuals. Use a numerically stabilized Gaussian copula; no cross-sensor dependence is modeled.
- Evaluate exactly the three path baselines requested in the lean Phase 4 prompt: independent Chronos quantile samples, Chronos samples with pooled global copula, and a 50/50 global label-free BMA sample (Phase 2 calibration-selected global weight). No external code or package is reused.
- For the D8 word “similar,” use an explicit ±5 percentage-point transition-coverage tolerance against the consecutive-origin ACI figure; compare the FORK mean 60-minute interval width against 18.39 mph. This is a descriptive rule, not a tuning target.
- Energy score uses a fixed-seed 500-pair Monte Carlo estimate; variogram score uses p=0.5 across all horizon pairs. Branch forecasts use 200 paths and deterministic seeds.
- Full Phase 4 experiment was intentionally not run by the agent per the user's final instruction; selected calibration values and test numbers remain pending the user's one-command run.

### Phase 3 approved implementation choices

- Use cyclical time-of-day and day-of-week features encoded as sine/cosine pairs from each window origin; normalize sensor readings with training-block sensor mean/std only. Scale each of the four calendar coordinates by sqrt(N/4) so their combined squared distance is comparable to the N normalized context coordinates.
- Reuse WindowMatcher, adding optional calendar features and an explicit exclusion radius. During detector training, exclude candidate origins within 180 five-minute rows (15 hours) of the query origin. Candidate contexts and futures must still lie wholly in the training block.
- Sweep k in [25, 50, 100]; choose k and logistic L2 strength on calibration by log loss for the approximate-90 Chronos miss target only. Do not use transition/regime labels for detector training or tuning.
- Use deterministic 2-means on train-standardized 12-step neighbor futures. Bimodality separation is Euclidean centroid distance divided by pooled within-cluster RMS spread; minority share is min(cluster sizes)/k, retained as the Phase 4 branch weight.
- Implement the small logistic regression and 2-means in NumPy to avoid new dependencies. Standardize detector features using training statistics. The supervised target is approximate-90 Chronos miss, not a transition label; call its output p(fork) and evaluate it against transition labels only after all choices are frozen.
- Tune volatility, recent-drop, and negative CUSUM thresholds on calibration using the Chronos-miss target; use fixed CUSUM reference 0.5 training standard deviations. Evaluate PR/AUPRC against 60-minute transition labels on test only.
- Set the hard-flag threshold to the calibration p(fork) quantile that yields the calibration Chronos-miss fraction; freeze it before test and report test flag rate.
- Generate synthetic independent 14-hour contexts with two speed regimes, observable fork-odds levels, configurable jump magnitude and branch coupling. Fit/use the PEMS-trained detector without exposing synthetic fork outcomes; compare predicted p(fork) to generator-known odds as a scoring check only.
- No code, tuning decision, or feature may inspect test transition labels. Test transition labels are passed only to final scoring and plotting functions.

### Phase 3 measured selections and review

- Selected k=25 and logistic L2=0.01 by calibration log loss on the approximate-90 miss target; calibration misses were 15.96%. Frozen hard-flag threshold was p(fork)=0.168745; test flag rate was 18.67%.
- Test AUPRCs for 60-minute transition: p(fork) 0.4299, volatility 0.4438, recent-drop 0.4527, CUSUM 0.3405. p(fork) does not beat volatility, so the requested kill/pivot signal is active. Calm false-alarm rates at calibration-frozen thresholds: p(fork) 0.123, volatility 0.874, recent-drop 0.186, CUSUM 1.000.
- Synthetic (600 episodes, 1,200 sensor windows; known mean odds 0.442, jump 24 mph, coupling 0.8): mean predicted p=0.609, odds-bin MAE 0.310, Brier against known fractional odds 0.239, AP against realized fork events 0.387 (realized event rate 0.471). This is poor probability transfer.
- Interpretation: this detector is not validated as a transition detector and its output is trained to predict model misses, not a directly observed two-branch event. Keep it as an experimental risk score; revisit the miss proxy/features before treating p(fork) as calibrated latent fork probability.
