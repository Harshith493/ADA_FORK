# Phase 5 path coverage audit

The old `whole_path_coverage.csv` `whole_path_region_coverage` is coverage of the max-score region: one common conformal level is applied across all 12 horizons, and the path is covered when every realized step lies inside that region. `joint_12_interval_coverage` is the distinct event that each of the 12 separately calibrated marginal intervals covers its step. They are not interchangeable.

There was no metric arithmetic bug. The max-score branch nonconformity was capped at 1; at level 1 the branch interval is limited to the finite empirical sample support. Outcomes beyond that support remain misses, explaining FORK's ~66.3%/66.4% group path-region coverage. The new uncapped Chronos scaled score can expand beyond the original P10-P90 band; its path-region coverage is 89.5% by p(fork) group and 89.8% without a gate. Its joint 12-marginal coverage remains 63.1% calm / 75.7% forked (69.7% no gate), as expected for a stricter event.

The old Chronos ACI rows used the same bounded sample-support score and therefore could not widen beyond the native band. They are relabelled “capped, not a valid baseline.” The replacement score is `abs(y-q50)/max((q90-q10)/2, 1e-3)` with bounds `q50 +/- tau*halfwidth`; it is unbounded and the level is calibrated online.
