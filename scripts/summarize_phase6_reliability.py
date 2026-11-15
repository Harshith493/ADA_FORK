"""Regenerate the 5 fixed-width minority branch reliability bins from cache."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase6 import minority_reliability
with np.load(ROOT/'data/cache/phase4_branch_cache.npz') as z: w,c,p=z['weights'],z['branch_counts'],z['branch_paths']
with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz') as z: y=z['actuals'].reshape(-1,12)
rows,brier,pred,hit=minority_reliability(w,c,p,y,5); df=pd.DataFrame(rows,columns=['mean_minority_weight','minority_nearest_frequency','n']); df.to_csv(ROOT/'outputs/phase6/branch_reliability.csv',index=False)
fig,ax=plt.subplots(figsize=(5,4)); ax.plot([0,.5],[0,.5],'--',color='gray'); ax.plot(df.mean_minority_weight,df.minority_nearest_frequency,'o-'); ax.set(xlabel='stated minority weight',ylabel='observed nearest-minority frequency',title=f'Brier={brier:.3f}'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(ROOT/'outputs/phase6/branch_reliability.png',dpi=150); plt.close(fig)
sel=(pred>=.25)&(pred<=.35); print(f'Brier={brier:.4f}; n in .25-.35={sel.sum()}; observed frequency={hit[sel].mean():.4f}')
