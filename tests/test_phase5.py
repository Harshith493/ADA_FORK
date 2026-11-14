from __future__ import annotations
from pathlib import Path
import ast, sys, unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase5 import (ConformalState,MaturityQueue,branch_bounds,
    branch_nonconformity,finite_sample_level)

class Phase5Tests(unittest.TestCase):
    def test_branch_score_and_full_support_cap(self):
        p=np.stack([np.tile(np.arange(5.)[:,None],(1,3)),np.tile((np.arange(5.)+10)[:,None],(1,3))])
        counts=np.array([5,5]); weights=np.array([.8,.2]); actual=np.array([2.,12.,30.])
        scores=branch_nonconformity(p,counts,weights,actual,.1,20)
        self.assertLessEqual(scores[0],1.); self.assertLessEqual(scores[1],1.); self.assertEqual(scores[2],1.)
        lo,hi=branch_bounds(p,counts,weights,1.,.1)
        self.assertEqual((lo[0],hi[0]),(0.,14.))

    def test_group_states_are_independent(self):
        state=ConformalState(4,.1)
        state._group(0)
        state.update(1,0,.8,1,.02)
        self.assertAlmostEqual(state.states[1]['alpha'][0],.082)
        self.assertAlmostEqual(state.states[0]['alpha'][0],.1)
        self.assertEqual(len(state.states[1]['buffers'][0]),1)
        self.assertEqual(len(state.states[0]['buffers'][0]),0)

    def test_only_matured_outcomes_update_state(self):
        state=ConformalState(1,.1); queue=MaturityQueue(); observed=[]
        queue.push(15,0,0,reference=7,bounds=(0.,1.))
        observe=lambda ref,stream,bounds:(observed.append((ref,stream)) or (.7,1))
        self.assertEqual(queue.mature(14,state,.01,observe=observe),[])
        self.assertEqual(observed,[])
        self.assertEqual(len(state._group(0)['buffers'][0]),0)
        self.assertEqual(queue.mature(15,state,.01,observe=observe),[15])
        self.assertEqual(observed,[(7,0)])
        self.assertAlmostEqual(state.states[0]['alpha'][0],.091)

    def test_random_gate_stream_is_near_nominal_and_deterministic(self):
        def run(seed):
            rng=np.random.default_rng(seed); state=ConformalState(1,.1); queue=MaturityQueue(); flags=rng.integers(0,2,12000)
            covered=[]
            for i,flag in enumerate(flags):
                queue.mature(i,state,.01)
                group=int(flag); tau=state.level(group,0,.1); score=float(rng.uniform()); err=score>tau
                covered.append(not err); queue.push(i+1,group,0,score,err)
            return np.asarray(covered)
        a=run(83); b=run(83); np.testing.assert_array_equal(a,b)
        self.assertLess(abs(a.mean()-.9),.03)

    def test_conformal_method_module_is_label_free(self):
        source=(ROOT/'src/fork/phase5.py').read_text(encoding='utf-8'); tree=ast.parse(source)
        imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('phase1' in x or 'label_target_period' in x for x in imports))
        self.assertNotIn('label_target_period',source)

if __name__=='__main__': unittest.main()
