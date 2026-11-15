"""Assemble a compact replay cache from existing local forecasts and branches."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase1 import _load_period,_split_indices,label_target_period

def main():
 p1=json.loads((ROOT/'configs/phase1_pems_bay.json').read_text()); p3=json.loads((ROOT/'configs/phase3_pems_bay.json').read_text())
 frame,columns=_load_period(ROOT/p1['dataset_path'],p1['period_start'],p1['period_end_exclusive']); values=frame[columns].to_numpy(np.float32); names={str(s):i for i,s in enumerate(columns)}; context=p1['context_steps']; train_end,_=_split_indices(len(frame),p1['train_fraction'],p1['calibration_fraction'])
 with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz') as z: qs,ys,origins,sensors=z['forecasts'],z['actuals'],z['origins'],z['sensors']
 with np.load(ROOT/'data/cache/phase4_branch_cache.npz') as z: paths,counts,weights=z['branch_paths'],z['branch_counts'],z['weights']
 with np.load(ROOT/p3['fork_output_cache']) as z: p_fork=z['p_fork_test']
 q=qs.reshape(-1,12,5); y=ys.reshape(-1,12); origin_rows=np.tile(origins,3*len(sensors[0])); group=np.repeat(np.arange(3),len(sensors[0])*len(origins)); slot=np.tile(np.repeat(np.arange(len(sensors[0])),len(origins)),3)
 ctx=np.empty((len(y),context),np.float32); labels=[]; train_min=np.empty(len(y),np.float32); train_max=np.empty(len(y),np.float32)
 for i,(g,s,o) in enumerate(zip(group,slot,origin_rows)):
  col=names[str(sensors[g,s])]; ctx[i]=values[int(o)-context:int(o),col]; train_min[i]=values[:train_end,col].min(); train_max[i]=values[:train_end,col].max()
  labels.append(label_target_period(y[i],p1['free_flow_mph'],p1['congested_mph']))
 pflag=p_fork>=.168745
 dest=ROOT/'data/cache/phase7_demo_cache.npz'; np.savez_compressed(dest,context=ctx,quantiles=q,actual=y,branch_paths=paths,branch_counts=counts,branch_weights=weights,origins=origin_rows,group=group,slot=slot,sensors=sensors[group,slot],regime=np.asarray(labels),p_fork=p_fork,pflag=pflag,training_min=train_min,training_max=train_max)
 print(f'Created local replay cache: {len(y)} windows; no inference or downloads.')
if __name__=='__main__': main()
