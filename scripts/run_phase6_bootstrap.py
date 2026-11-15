"""Recompute deterministic day-block CIs from the stored alpha=.1 replay."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase1 import _load_period
from fork.phase6 import day_block_bootstrap_ci
cfg=json.loads((ROOT/'configs/phase1_pems_bay.json').read_text()); frame,_=_load_period(ROOT/cfg['dataset_path'],cfg['period_start'],cfg['period_end_exclusive'])
with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz') as z: origins=z['origins']
day={int(o):pd.Timestamp(frame.timestamp.iloc[int(o)]).date().toordinal() for o in origins}; df=pd.read_csv(ROOT/'outputs/phase6/matched_curves.csv'); df=df[(df.alpha==.1)&(df.horizon==60)].copy(); df['day']=df.origin.map(day); days=sorted(df.day.unique()); lookup={d:i for i,d in enumerate(days)}
rows=[]
for method,g in df.groupby('method'):
 ids=g.day.map(lookup).to_numpy(); transition=g.regime.eq('transition').to_numpy(); calm=g.calm.to_numpy(); fork=~calm; point=g.fork_abs_error.to_numpy(); chron=g.chronos_abs_error.to_numpy()
 specs={'transition_coverage':(transition,g.covered.to_numpy()),'calm_group_coverage':(calm,g.covered.to_numpy()),'forked_group_coverage':(fork,g.covered.to_numpy()),'transition_width':(transition,g.width.to_numpy()),'transition_interval_score':(transition,g.interval_score.to_numpy()),'weighted_median_point_mae':(np.ones(len(g),bool),point),'chronos_q50_point_mae':(np.isfinite(chron),chron)}
 ci=day_block_bootstrap_ci(ids,specs,len(days),1000,6060)
 for name,(lo,hi) in ci.items():
  mask,val=specs[name]; subset='all test windows at 60m' if 'point_mae' in name else 'transition windows' if name.startswith('transition_') else 'gate-flagged group'; rows.append({'method':method,'metric':name,'subset':subset,'estimate':float(np.mean(val[mask])),'ci95_low':lo,'ci95_high':hi})
pd.DataFrame(rows).to_csv(ROOT/'outputs/phase6/day_block_bootstrap_ci.csv',index=False,float_format='%.6f')
print('Wrote deterministic 1000-resample day-block CIs.')
