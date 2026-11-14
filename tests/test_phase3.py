from __future__ import annotations
from pathlib import Path
import ast
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fork.window_matching import WindowMatcher
from fork.phase3 import (baseline_scores,binary_log_loss,calendar_features,
    detector_features,fit_logistic,generate_synthetic,kmeans_2,predict_logistic,
    precision_recall_curve)


class Phase3DetectorTests(unittest.TestCase):
    def test_matcher_excludes_own_time_neighborhood_and_stays_in_train(self):
        values=np.sin(np.arange(1200)/17).astype(np.float32)
        origins=np.arange(168,1188-12,12)
        times=np.tile(np.array([0.,1.,0.,1.],np.float32),(len(origins),1))
        matcher=WindowMatcher(values,168,12,origins,times)
        qorigin=int(origins[20])
        qtime=times[20]
        _,found=matcher.query(values[qorigin-168:qorigin],10,qtime,
                              exclude_origin=qorigin,exclusion_radius=180)
        self.assertTrue(np.all(np.abs(found-qorigin)>180))
        self.assertTrue(np.all(found+12<=len(values)))

    def test_kmeans_detects_two_separated_path_groups(self):
        paths=np.concatenate([np.zeros((30,12)),np.full((20,12),5.)])
        labels,centers,gap,share,score=kmeans_2(paths,seed=13)
        self.assertEqual(set(np.unique(labels)),{0,1})
        self.assertAlmostEqual(share,.4)
        self.assertGreater(gap,10)
        self.assertGreater(score,1000)

    def test_detector_features_and_logistic_are_deterministic(self):
        paths=np.concatenate([np.zeros((25,12)),np.full((25,12),8.)])
        q=np.tile(np.array([1,2,4,6,7.]),(12,1))
        f=detector_features(q,paths,np.zeros(168),30,5,seed=17)
        self.assertEqual(f.shape,(4,))
        x=np.vstack([f,f+[1,0,1,1],f+[0,.2,0,.2],f+[2,0,2,1]])
        y=np.array([0,1,0,1])
        m1=fit_logistic(x,y,l2=.1); m2=fit_logistic(x,y,l2=.1)
        np.testing.assert_allclose(predict_logistic(m1,x),predict_logistic(m2,x))

    def test_detector_module_has_no_transition_label_dependency(self):
        source=(ROOT/'src/fork/phase3.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                imported.append(ast.unparse(node))
        self.assertFalse(any('phase1' in item for item in imported))
        self.assertNotIn('label_target_period',source)

    def test_calendar_features_are_cyclical_and_four_dimensional(self):
        features=calendar_features(np.datetime64('2017-01-02T12:00'))
        self.assertEqual(features.shape,(4,))
        self.assertAlmostEqual(float(features[0]),0.,places=6)
        self.assertAlmostEqual(float(features[1]),-1.,places=6)

    def test_synthetic_generator_is_fixed_seed_with_known_marginal_odds(self):
        a=generate_synthetic(n=40,seed=33); b=generate_synthetic(n=40,seed=33)
        np.testing.assert_array_equal(a['contexts'],b['contexts'])
        np.testing.assert_array_equal(a['fork_odds'],b['fork_odds'])
        self.assertEqual(a['contexts'].shape,(40,2,168))
        self.assertEqual(a['futures'].shape,(40,2,12))

    def test_precision_recall_and_logloss_metrics(self):
        p,r,thresholds,ap=precision_recall_curve([0,1,0,1],[.1,.9,.2,.8])
        self.assertEqual(len(p),5); self.assertEqual(len(r),5)
        self.assertEqual(ap,1.)
        self.assertLess(binary_log_loss([0,1],[.1,.9]),.2)

    def test_precision_recall_groups_tied_scores_independent_of_row_order(self):
        _,_,_,a=precision_recall_curve([1,0,0,1],[.5,.5,.1,.1])
        _,_,_,b=precision_recall_curve([0,1,1,0],[.1,.5,.1,.5])
        self.assertAlmostEqual(a,b)

    def test_logistic_fit_api_has_no_transition_label_parameter(self):
        import inspect
        self.assertEqual(list(inspect.signature(fit_logistic).parameters)[:2],['features','misses'])


if __name__=='__main__': unittest.main()
