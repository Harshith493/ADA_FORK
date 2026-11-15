"""Fast, cache-only matched calm-coverage comparisons after Phase 6 replay."""
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'outputs/phase6'
sys.path.insert(0,str(ROOT/'src'))
from fork.phase6 import interpolate_at
df=pd.read_csv(out/'coverage_width_curve.csv'); rows=[]
for h in (15,30,60):
 g=df[(df.horizon==h)&(df.subset=='calm')]
 for target in (.90,):
  vals={m:(interpolate_at(x.coverage,x.width,target),interpolate_at(x.coverage,x.interval_score,target)) for m,x in g.groupby('method')}
  f=vals.get('FORK ACI (p(fork))',(None,None)); c=vals.get('Chronos scaled ACI (p(fork) gate)',(None,None))
  rows.append({'horizon':h,'target_calm_coverage':target,'fork_width':f[0],'chronos_width':c[0],'width_reduction':(c[0]-f[0])/c[0] if f[0] is not None and c[0] not in (None,0) else None,'fork_interval_score':f[1],'chronos_interval_score':c[1]})
pd.DataFrame(rows).to_csv(out/'matched_calm_coverage.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
