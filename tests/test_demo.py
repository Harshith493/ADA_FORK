import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

class StreamlitDemoSmokeTests(unittest.TestCase):
    def test_app_imports_and_loads_one_cached_window(self):
        from app.demo import load_demo_data, choose_window
        cache=ROOT/'data/cache/phase7_demo_cache.npz'
        if not cache.exists(): self.skipTest('Run scripts/build_phase7_demo_cache.py to create the ignored local replay cache.')
        data=load_demo_data(cache); index=choose_window(data,'Index',0)
        self.assertEqual(index,0); self.assertEqual(data['context'][index].shape,(168,)); self.assertEqual(data['actual'][index].shape,(12,))

if __name__=='__main__': unittest.main()
