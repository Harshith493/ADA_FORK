"""Lean Phase 4 scenario branch experiment; all construction is label-free."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fork.phase1 import _load_period,_split_indices,label_target_period,QUANTILES,approximate_90_bounds
from fork.phase2 import interval_score,quantiles_to_samples
from fork.phase3 import calendar_features
from fork.window_matching import WindowMatcher
from fork.phase4 import (cluster_branches,corr_from_residuals,shrink_correlation,
    quantile_copula_paths,sample_branch_paths,split_interval,energy_score,variogram_score)

def write_markdown_table(df,path):
    columns=list(df.columns)
    def cell(value):
        if isinstance(value,(float,np.floating)): return f'{value:.4f}'
        return str(value).replace('|','\\|')
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(cell(row[c]) for c in columns)+' |' for _,row in df.iterrows())
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/phase4_pems_bay.json'); args=ap.parse_args()
    cfg=json.loads((ROOT/args.config).read_text()); p1=json.loads((ROOT/cfg['phase1_config']).read_text()); p3=json.loads((ROOT/cfg['phase3_config']).read_text())
    frame,cols=_load_period(ROOT/p1['dataset_path'],p1['period_start'],p1['period_end_exclusive']); values=frame[cols].to_numpy(np.float32); times=frame.timestamp
    tr_end,cal_end=_split_indices(len(frame),p1['train_fraction'],p1['calibration_fraction'])
    with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz',allow_pickle=False) as a: tq,ty,to,sensors=a['forecasts'],a['actuals'],a['origins'],a['sensors']
    with np.load(ROOT/'data/cache/phase2_calib_forecasts.npz',allow_pickle=False) as a: cq,cy,co=a['forecasts'],a['actuals'],a['origins']
    with np.load(ROOT/p3['train_forecast_cache'],allow_pickle=False) as a: trq,try_,trs,tro=a['forecasts'],a['actuals'],a['sensors'],a['origins']
    sensor_idx={s:i for i,s in enumerate(cols)}; train_origins=np.arange(p1['context_steps'],tr_end-12+1)
    cal_times=np.stack([calendar_features(pd.Timestamp(times.iloc[int(o)])) for o in train_origins])
    selected_sensors=sorted(set(sensors.ravel().tolist())); matchers={}
    for s in selected_sensors:
        matchers[s]=WindowMatcher(values[:tr_end,sensor_idx[s]],168,12,train_origins,candidate_time_features=cal_times)
    # One pooled train-only residual correlation, from cached Chronos train forecasts.
    pooled=corr_from_residuals(try_-trq[:,:,2]); out=ROOT/cfg['output_dir']; out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/cfg['branch_cache']; cal_records=[]; candidates=[]
    for name,qs,ys,origins in [('cal',cq,cy,co),('test',tq,ty,to)]:
        path_rows=[]; tune_n=cfg['calibration_tune_origins'] if name=='cal' else len(origins)
        for g in range(3):
          for si,s in enumerate(sensors[g]):
            matcher=matchers[s]; mean=matcher.mean; std=matcher.std
            for wi,o in enumerate(origins[:tune_n] if name=='cal' else origins):
              q=qs[g,si,wi]; y=ys[g,si,wi]; ctx=values[int(o)-168:int(o),sensor_idx[s]]
              tf=calendar_features(pd.Timestamp(times.iloc[int(o)])); hist,found=matcher.query(ctx,cfg['k'],tf,exclude_origin=int(o),exclusion_radius=180)
              branch=cluster_branches(hist,q[:,2],mean,std,seed=cfg['seed']+wi+si)
              if name=='cal':
                record={'q':q,'actual':y,'branch':branch,'wi':wi}; cal_records.append(record)
                # Tune on calibration outcomes only and retain scalar losses, not sample arrays.
                for alpha in cfg['alpha_grid']:
                  for threshold in cfg['threshold_grid']:
                    _,parts,weights,_=sample_branch_paths(branch,q,pooled,alpha,threshold,.5,cfg['paths'],cfg['seed']+wi,True)
                    lo,hi=split_interval(parts,weights,threshold,.1)
                    candidates.append((alpha,threshold,interval_score(y[11],lo[11],hi[11],.1)))
    candidate_means=[]
    for alpha in cfg['alpha_grid']:
      for threshold in cfg['threshold_grid']:
        candidate_means.append((float(np.mean([z for a,t,z in candidates if a==alpha and t==threshold])),alpha,threshold))
    _,best_alpha,best_th=min(candidate_means)
    pd.DataFrame(candidate_means,columns=['calibration_interval_score','alpha','branch_threshold']).to_csv(out/'calibration_sweep.csv',index=False)
    # Tune copula shrinkage on the same calibration-only sample and selected branch settings.
    shrink_scores=[]
    for shrink in cfg['shrinkage_grid']:
        vals=[]
        for r in cal_records:
            paths,_,_,_=sample_branch_paths(r['branch'],r['q'],pooled,best_alpha,best_th,shrink,cfg['paths'],cfg['seed']+r['wi'],True)
            vals.append(energy_score(paths,r['actual'],cfg['seed']+r['wi'],cfg['energy_pair_count']))
        shrink_scores.append((float(np.mean(vals)),float(shrink)))
    best_shrink=min(shrink_scores)[1]
    pd.DataFrame(shrink_scores,columns=['calibration_energy_score','shrinkage']).to_csv(out/'shrinkage_sweep.csv',index=False)
    # Rebuild frozen test rows with selected settings; test labels only enter scoring below.
    chosen=[]; rows=[]
    for g in range(3):
      for si,s in enumerate(sensors[g]):
       matcher=matchers[s]
       for wi,o in enumerate(to):
        q=tq[g,si,wi]; y=ty[g,si,wi]; ctx=values[int(o)-168:int(o),sensor_idx[s]]; tf=calendar_features(pd.Timestamp(times.iloc[int(o)]))
        hist,_=matcher.query(ctx,cfg['k'],tf,exclude_origin=int(o),exclusion_radius=180); branch=cluster_branches(hist,q[:,2],matcher.mean,matcher.std,seed=cfg['seed']+wi+si)
        fork,parts,weights,_=sample_branch_paths(branch,q,pooled,best_alpha,best_th,best_shrink,cfg['paths'],cfg['seed']+wi+g*100+si,True)
        chron_iid=quantiles_to_samples(q,QUANTILES,cfg['paths']); chron_cop=quantile_copula_paths(q,QUANTILES,pooled,cfg['paths'],cfg['seed']+wi)
        bma=np.concatenate([chron_iid[:cfg['paths']//2],hist[np.random.default_rng(cfg['seed']+wi).integers(len(hist),size=cfg['paths']//2)]])
        methods={'FORK branches':(fork,*split_interval(parts,weights,best_th,.1)), 'Chronos independent':(chron_iid,*np.quantile(chron_iid,[.05,.95],axis=0)), 'Chronos global copula':(chron_cop,*np.quantile(chron_cop,[.05,.95],axis=0)), 'Global label-free BMA':(bma,*np.quantile(bma,[.05,.95],axis=0))}
        for method,(paths,lo,hi) in methods.items():
         label=label_target_period(y,p1['free_flow_mph'],p1['congested_mph'])
         for h in (2,5,11): rows.append({'method':method,'regime':label,'horizon_minutes':(h+1)*5,'actual':float(y[h]),'lower':float(lo[h]),'upper':float(hi[h]),'interval_score':interval_score(y[h],lo[h],hi[h],.1)})
         chosen.append((method,label,energy_score(paths,y,cfg['seed']+wi,cfg['energy_pair_count']),variogram_score(paths,y,.5)))
        path_rows.append((g,si,wi,parts,weights,q))
    df=pd.DataFrame(rows); pathdf=pd.DataFrame(chosen,columns=['method','regime','energy_score','variogram_score'])
    metrics=df.groupby(['method','regime','horizon_minutes']).apply(lambda x:pd.Series({'n':len(x),'coverage':np.mean((x.actual>=x.lower)&(x.actual<=x.upper)),'mean_width_mph':np.mean(x.upper-x.lower),'interval_score':x.interval_score.mean()}),include_groups=False).reset_index()
    scores=pathdf.groupby(['method','regime']).mean(numeric_only=True).reset_index(); metrics=metrics.merge(scores,on=['method','regime'],how='left')
    metrics.to_csv(out/'results.csv',index=False,float_format='%.6f'); write_markdown_table(metrics,out/'results.md')
    headline=metrics[metrics.horizon_minutes==60].copy()
    headline.to_csv(out/'headline_60m.csv',index=False,float_format='%.6f')
    f=headline[headline.method=='FORK branches']; tr=f[f.regime=='transition']; calm=f[f.regime!='transition']
    if len(tr) and len(calm):
        close=abs(float(tr.coverage.iloc[0])-.582)<=.05
        calm_cov=float(np.average(calm.coverage,weights=calm.n))
        overall_width=float(np.average(f.mean_width_mph,weights=f.n))
        sharper=overall_width<18.39
        verdict=f"Against consecutive-origin ACI (transition coverage 58.2%, calm coverage 91.7%, width 18.39 mph), FORK transition coverage is {float(tr.coverage.iloc[0]):.1%}, calm-group coverage is {calm_cov:.1%}, and mean 60-minute width is {overall_width:.2f} mph.\n\n"
        verdict+=('Coverage is within the documented ±5 percentage-point similarity band. ' if close else 'Coverage is outside the documented ±5 percentage-point similarity band. ')
        verdict+=('FORK is sharper by mean width.\n' if sharper else 'FORK is not sharper by mean width.\n')
    else: verdict='D8 comparison unavailable: no transition or calm rows in the test metrics.\n'
    (out/'D8_verdict.md').write_text('## Phase 2 consecutive-origin ACI comparison\n\n'+verdict,encoding='utf-8')
    # Small visual audit: first test window, branch-weighted paths and approximate Chronos band.
    g,si,wi,branch_paths,w,q=path_rows[0]; fp=np.concatenate(list(branch_paths.values())); lo,hi=approximate_90_bounds(q); fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(1,13)
    ax.fill_between(x,lo,hi,color='gray',alpha=.22,label='Chronos approx 90%'); ax.plot(x,q[:,2],color='black',label='Chronos median')
    colors=['tab:blue','tab:orange']
    for j,paths in branch_paths.items():
        ax.plot(x,np.median(paths,axis=0),color=colors[j],label=f'Branch {j+1} (weight {w[j]:.2f})')
    ax.fill_between(x,np.quantile(fp,.1,axis=0),np.quantile(fp,.9,axis=0),color='tab:blue',alpha=.12,label='FORK total 10–90%')
    ax.set(xlabel='5-minute step',ylabel='Speed (mph)',title='Example test window: Chronos and FORK paths'); ax.legend(); fig.tight_layout(); fig.savefig(out/'example_branches.png',dpi=160); plt.close(fig)
    branches=np.full((len(path_rows),2,cfg['paths'],12),np.nan,np.float32); counts=np.zeros((len(path_rows),2),np.int16)
    for i,item in enumerate(path_rows):
        for j,paths in item[3].items(): counts[i,j]=len(paths); branches[i,j,:len(paths)]=paths
    np.savez_compressed(cache,metadata=np.array(json.dumps({'alpha':best_alpha,'threshold':best_th,'shrinkage':best_shrink,'k':cfg['k'],'paths':cfg['paths'],'origins':to.tolist(),'sensors':sensors.tolist()})),weights=np.stack([x[4] for x in path_rows]),branch_paths=branches,branch_counts=counts)
    print(f'Phase 4 outputs: {out}; selected alpha={best_alpha}, threshold={best_th}, shrinkage={best_shrink}; rows={len(metrics)}')

if __name__=='__main__': main()
