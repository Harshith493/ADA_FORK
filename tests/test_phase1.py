from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fork.phase1 import (  # noqa: E402
    _generate_forecasts,
    approximate_90_bounds,
    coverage_rate,
    label_target_period,
)


class CoverageMetricTests(unittest.TestCase):
    def test_coverage_is_inclusive_and_averaged(self) -> None:
        self.assertAlmostEqual(
            coverage_rate(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 0.0, 3.0]),
                np.array([1.0, 1.0, 4.0]),
            ),
            2 / 3,
        )

    def test_coverage_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            coverage_rate(np.array([1.0]), np.array([0.0, 1.0]), np.array([2.0]))

    def test_coverage_rejects_reversed_bounds(self) -> None:
        with self.assertRaises(ValueError):
            coverage_rate(np.array([1.0]), np.array([2.0]), np.array([0.0]))


class TransitionLabelTests(unittest.TestCase):
    def test_all_fast_period_is_free_flow(self) -> None:
        self.assertEqual(label_target_period(np.array([56.0, 65.0])), "free-flow")

    def test_all_slow_period_is_congested(self) -> None:
        self.assertEqual(label_target_period(np.array([10.0, 24.0])), "congested")

    def test_mixed_or_threshold_values_are_transition(self) -> None:
        self.assertEqual(label_target_period(np.array([65.0, 40.0])), "transition")
        self.assertEqual(label_target_period(np.array([55.0, 60.0])), "transition")
        self.assertEqual(label_target_period(np.array([24.0, 25.0])), "transition")


class LabelIsolationTests(unittest.TestCase):
    def test_only_past_context_reaches_forecaster(self) -> None:
        class SpyForecaster:
            def __init__(self) -> None:
                self.seen = None

            def forecast(self, contexts: np.ndarray) -> np.ndarray:
                self.seen = contexts.copy()
                return np.zeros((contexts.shape[0], 3, 5), dtype=np.float32)

        timestamps = pd.date_range("2017-01-01", periods=10, freq="5min")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "A": np.arange(10, dtype=np.float32),
                "B": np.arange(100, 110, dtype=np.float32),
            }
        )
        spy = SpyForecaster()
        _generate_forecasts(
            frame,
            ["A", "B"],
            [["A", "B"]],
            np.array([5]),
            context_steps=2,
            max_horizon=3,
            forecaster=spy,
        )
        np.testing.assert_array_equal(spy.seen, np.array([[3.0, 4.0], [103.0, 104.0]], dtype=np.float32))
        self.assertFalse(np.isin(spy.seen, [5.0, 6.0, 7.0, 105.0, 106.0, 107.0]).any())


class IntervalConstructionTests(unittest.TestCase):
    def test_outer_tails_are_linearly_extrapolated(self) -> None:
        q = np.array([10.0, 20.0, 50.0, 80.0, 90.0])
        lower, upper = approximate_90_bounds(q)
        self.assertEqual(float(lower), 5.0)
        self.assertEqual(float(upper), 95.0)


if __name__ == "__main__":
    unittest.main()
