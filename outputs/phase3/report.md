# Phase 3: label-free fork detector

- Detector and tuning code receive no transition labels. The logistic target is the training/calibration approximate-90 Chronos miss indicator; transition labels are created only for final test scoring. PEMS-BAY is historical (2017) and may overlap Chronos training data.
- Train/calibration/test row counts: 30,433/4,347/8,696; non-overlapping model training examples: 5,000; test score rows: 4,500.
- Forecast/training cache: hit; test forecasts reused from Phase 1; calibration forecasts reused from Phase 2. Feature cache: hit.
- Selected k=25, L2=0.01 by calibration miss log loss. Hard flag is the calibration p(fork) quantile matching calibration miss fraction 16.0%; threshold=0.16874; test flag rate=18.7%.

## Detector results: 60-minute transition target, test only

| Score | AUPRC | False alarm rate on calm | Test flag rate | Threshold selection |
|---|---:|---:|---:|---|
| p(fork) | 0.430 | 0.123 | 0.187 | calibration miss-rate quantile |
| volatility | 0.445 | 0.874 | 0.891 | calibration F1 vs miss |
| recent-drop | 0.453 | 0.186 | 0.256 | calibration F1 vs miss |
| CUSUM | 0.302 | 1.000 | 1.000 | calibration F1 vs miss |

Best baseline by test AUPRC is recent-drop (0.453); p(fork) is 0.430. Kill/pivot signal (p(fork) no better than volatility baseline): YES.
This signal means the fork score did not improve on volatility for transition ranking; do not claim that the detector is validated as a transition detector.

The output p(fork) is trained against Chronos interval misses as specified, so it is a fork-risk proxy, not a directly observed probability that the future has two latent branches. The synthetic known-odds check tests calibration transfer but does not redefine the supervised target.

## Synthetic regime-switching check

Two speed regimes, jump=24.0 mph, branch coupling=0.80, 600 episodes/1200 sensor windows. Known mean fork odds=0.442; mean p(fork)=0.609; odds-bin MAE=0.310; Brier vs odds=0.239; AP vs realized fork events=0.387.

## Reproducibility and leakage

- Seeded training-window sample 3051; feature seed 3117; synthetic seed 90210. Training windows are sampled on a 180-row stride per sensor, then seeded subsampled.
- Historical neighbor search is training-only. Self-neighborhood exclusion is ±180 rows. Context normalization and calendar features use no test outcomes.
- Four cyclical calendar coordinates are scaled by sqrt(168/4), balancing their combined distance contribution against the 168 normalized readings.
- k/L2 and baseline thresholds use calibration miss targets only. Test transition labels are used only to score the frozen detector/baseline scores and false alarms.
- Cached p(fork) and minority shares for Phase 4/5: data/cache/phase3_fork_outputs.npz. Detailed table/PR points and plot data are saved in outputs/phase3/.
- Synthetic detector scores use the PEMS-trained model; generator fork outcomes and known odds are held back until scoring. It is a transfer stress test, not a claim of synthetic probability calibration.

No new packages or external code were used.
