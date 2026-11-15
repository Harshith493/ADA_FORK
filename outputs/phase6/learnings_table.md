# Expected learnings: Phase 6 evidence

The compact one-pager summary does not reproduce the full original “Expected Learnings” table. This table therefore records every explicit target and kill/pivot criterion available in the summary and Phase 6 request; omitted original rows are not inferred.

| Learning / target | Measured evidence | Status |
|---|---|---|
| Reproduce Chronos-Bolt's ~54.9% transition coverage (nominal 90%, 60m) | Phase 1 measured value is in `outputs/phase1/`; historic PEMS-BAY is a reproduction only because training overlap is unresolved. | Report against paper; not post-cutoff evidence |
| History sampling ~81–82%; BMA+ACI 73.6% reference | Our Phase 2 values and gaps are in `outputs/phase2/report.md`; references are comparison points only. | Measured earlier; no retuning |
| Detector kill signal: p(fork) should beat simple risk baselines | p(fork) AUPRC 0.4299, volatility 0.4438, recent-drop 0.4527. | **Fail; active pivot signal** |
| 90% group split coverage for FORK | At alpha=.1: calm 88.4% (95% day CI 85.6–91.0), forked 89.3% (86.2–92.8). | Near target, point estimates below 90 |
| Calm-branch-collapse ablation | Calm-flagged coverage 82.9% after gamma grid extension; selected gamma was the new lower endpoint .001. | Fail; gamma boundary selection remains a convergence warning |
| 90% whole-path max-score region | FORK group coverage 66.3%/66.4%; finite branch support caps the score. | **Fail** |
| At least 85% 60m transition coverage | FORK 82.2% (95% day CI 79.0–86.3). | **Fail** |
| At least 10% sharper than Chronos at matched transition coverage | At 70% coverage, 60m interpolated width 27.54 vs 74.64 mph (63.1% lower) and interval score 43.29 vs 141.14. At 80%, Chronos scaled curve ends at 78.4%; no overlap. | Mixed; 70% point supports >10%, no 80% comparison |
| Branch minority odds: stated 30% occurs 25–35% | In predicted minority-weight [.25,.35], observed nearest-minority center frequency 33.1%; overall Brier 0.2141. | Pass for this bin; weak overall calibration |
| Phase 5 oracle gate | Only the Phase 5 gate uses true regime as a reference; no true-regime branch-construction oracle was run. | NOT DONE (branch oracle) |
| Sensor coupling | No cross-sensor copula/dependence. | NOT DONE |
| Other TSFMs | Chronos-Bolt only. | NOT DONE |
| Other datasets / METR-LA transfer | METR-LA download blocked by sandbox DNS; transfer caches/pipeline not built. | NOT DONE |
| fev-bench and GIFT-Eval | Not in MVP execution. | NOT DONE |
| Caltrans PeMS 2026 | Requires manual account and download. | NOT DONE |

Full measured tables and intervals are in `headline_60m.csv`, `day_block_bootstrap_ci.csv`, `matched_coverage.csv`, and `branch_reliability.csv` beside this file. Values are from 4,500 sampled test windows, not the all-origin scale-up.
