from __future__ import annotations
from pathlib import Path
import ast, inspect, sys, unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase4 import (active_branches,cluster_branches,gaussian_copula_uniforms,
    sample_branch_paths,smoothed_weights)
from fork.window_matching import WindowMatcher

class Phase4Tests(unittest.TestCase):
    def test_weights_sum_and_collapse(self):
        branch={'counts':np.array([24,1]),'centers':np.zeros((2,12)),
          'residuals':[np.zeros((24,12)),np.zeros((1,12))],'anchor':0}
        self.assertAlmostEqual(float(smoothed_weights([24,1],1).sum()),1.)
        active,w=active_branches(branch,1,.1)
        self.assertEqual(active,[0]); self.assertAlmostEqual(float(w.sum()),1.)

    def test_copula_shape_marginals_and_determinism(self):
        corr=np.full((3,3),.35); np.fill_diagonal(corr,1.)
        a=gaussian_copula_uniforms(corr,12000,7); b=gaussian_copula_uniforms(corr,12000,7)
        self.assertEqual(a.shape,(12000,3)); np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(np.abs(a.mean(axis=0)-.5)<.02))
        self.assertTrue(np.all(np.abs(a.std(axis=0)-np.sqrt(1/12))<.02))

    def test_branch_paths_are_deterministic_and_weighted(self):
        paths=np.concatenate([np.zeros((18,12)),np.full((7,12),8.)])
        branch=cluster_branches(paths,np.zeros(12),0.,1.,seed=2)
        q=np.tile([0.,1.,2.,3.,4.],(12,1)); pool=np.eye(12)
        a=sample_branch_paths(branch,q,pool,1.,0.,.5,200,10)
        b=sample_branch_paths(branch,q,pool,1.,0.,.5,200,10)
        np.testing.assert_array_equal(a[0],b[0]); self.assertEqual(a[0].shape,(200,12))
        self.assertAlmostEqual(float(a[2].sum()),1.)

    def test_labels_are_outside_branch_construction_and_tuning_api(self):
        source=(ROOT/'src/fork/phase4.py').read_text(encoding='utf-8'); tree=ast.parse(source)
        imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('phase1' in x or 'label_target_period' in x for x in imports))
        self.assertNotIn('label_target_period',source)
        self.assertNotIn('labels',inspect.signature(cluster_branches).parameters)
        self.assertNotIn('labels',inspect.signature(sample_branch_paths).parameters)

    def test_neighbor_match_excludes_query_neighborhood_and_uses_train_candidates(self):
        values=np.sin(np.arange(2200)/19).astype(np.float32)
        origins=np.arange(168,1500-12,15)
        times=np.tile(np.array([0.,1.,0.,1.],np.float32),(len(origins),1))
        matcher=WindowMatcher(values[:1500],168,12,origins,times)
        own=int(origins[20]); _,found=matcher.query(values[own-168:own],25,times[20],own,180)
        self.assertTrue(np.all(np.abs(found-own)>180)); self.assertTrue(np.all(found+12<=1500))

if __name__=='__main__': unittest.main()
