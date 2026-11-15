import ast, sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase6 import day_block_bootstrap_ci, interpolate_at, minority_reliability

class Phase6Tests(unittest.TestCase):
 def test_bootstrap_is_deterministic(self):
  ids=np.array([0,0,1,1,2,2]); spec={'x':(np.ones(6,bool),np.arange(6,dtype=float))}
  a=day_block_bootstrap_ci(ids,spec,3,100,123); b=day_block_bootstrap_ci(ids,spec,3,100,123); self.assertEqual(a,b)
 def test_interpolation_refuses_extrapolation(self):
  self.assertIsNone(interpolate_at([.6,.7],[1,2],.8)); self.assertAlmostEqual(interpolate_at([.6,.8],[1,3],.7),2.)
 def test_minority_reliability_drops_collapsed(self):
  paths=np.array([[[[0.],[0.]],[[3.],[3.]]],[[[0.],[0.]],[[0.],[0.]]]])
  rows,brier,pred,hit=minority_reliability(np.array([[.7,.3],[1.,0.]]),np.array([[2,2],[2,0]]),paths,np.array([[3.],[0.]]),5)
  self.assertEqual(len(pred),1); self.assertEqual(len(rows),5); self.assertAlmostEqual(float(hit[0]),1.); self.assertTrue(np.isfinite(brier))
 def test_phase6_helpers_do_not_import_scoring_labels(self):
  source=(ROOT/'src/fork/phase6.py').read_text(encoding='utf-8'); tree=ast.parse(source)
  imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
  self.assertFalse(any('phase1' in x or 'label_target_period' in x for x in imports)); self.assertNotIn('label_target_period',source)

if __name__=='__main__': unittest.main()
