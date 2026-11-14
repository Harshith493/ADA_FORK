from __future__ import annotations
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fork.phase2 import (aci_update, interval_score, mix_samples, quantiles_to_samples,
                         forecast_contexts)
from fork.window_matching import WindowMatcher, empirical_interval


class Phase2MetricTests(unittest.TestCase):
    def test_interval_score_width_and_miss_penalty(self):
        np.testing.assert_allclose(interval_score([1,0,3],[0,1,0],[2,2,2],.1),[2,21,22])

    def test_aci_update_rule(self):
        self.assertAlmostEqual(aci_update(.1,0,.01,.1),.101)
        self.assertAlmostEqual(aci_update(.1,1,.01,.1),.091)

    def test_history_interval_uses_empirical_quantiles(self):
        paths=np.arange(100,dtype=float)[:,None]
        lo,hi=empirical_interval(paths)
        self.assertAlmostEqual(lo[0],4.95)
        self.assertAlmostEqual(hi[0],94.05)

    def test_bma_mix_replaces_fixed_fraction(self):
        model=np.zeros((100,2)); hist=np.ones((100,2))
        mixed=mix_samples(model,hist,.2,np.random.default_rng(4))
        self.assertEqual(mixed.shape,(100,2))
        self.assertEqual(int(np.sum(mixed[:,0]==1)),20)

    def test_quantile_pseudo_samples_keep_horizon_axis(self):
        q=np.array([[0,1,2,3,4],[10,11,12,13,14]],float)
        samples=quantiles_to_samples(q,[.1,.2,.5,.8,.9],100)
        self.assertEqual(samples.shape,(100,2))
        self.assertLess(float(samples[:,1].mean()-samples[:,0].mean()),12)

    def test_forecaster_boundary_accepts_context_only(self):
        class Spy:
            def forecast(self,contexts):
                self.seen=contexts.copy()
                return np.zeros((len(contexts),12,5),np.float32)
        spy=Spy(); context=np.arange(168,dtype=np.float32)[None,:]
        forecast_contexts(spy,context)
        np.testing.assert_array_equal(spy.seen,context)
        self.assertFalse(np.isin(spy.seen,[168,169]).any())

    def test_matcher_uses_only_supplied_training_series(self):
        train=np.arange(30,dtype=np.float32)
        matcher=WindowMatcher(train,4,2)
        paths,_=matcher.query(train[-4:],3)
        self.assertTrue(np.all(paths<30))


if __name__=='__main__': unittest.main()
