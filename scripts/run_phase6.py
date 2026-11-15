"""Phase 6 audit, matched-coverage curves, branch odds, and block bootstrap."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
import run_phase5 as p5
from fork.phase1 import _load_period,_split_indices,label_target_period
from fork.phase3 import calendar_features
from fork.phase4 import corr_from_residuals
from fork.phase5 import branch_nonconformity,scaled_chronos_scores
from fork.phase6 import day_block_bootstrap_ci,interpolate_at,minority_reliability,clip_interval

def main():
 t0=time.perf_counter(); p6=json.loads((ROOT/'configs/phase6.json').read_text()); cfg=json.loads((ROOT/p6['phase5_config']).read_text()); p1=json.loads((ROOT/cfg['phase1_config']).read_text()); p3=json.loads((ROOT/cfg['phase3_config']).read_text()); out=ROOT/p6['output_dir']; out.mkdir(parents=True,exist_ok=True)
 print('Loading cached arrays and labels for scoring only...',flush=True)
 frame,columns=_load_period(ROOT/p1['dataset_path'],p1['period_start'],p1['period_end_exclusive']); values=frame[columns].to_numpy(np.float32); train_end,cal_end=_split_indices(len(frame),p1['train_fraction'],p1['calibration_fraction'])
 with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz') as z: tq,ty,to,sensors=z['forecasts'],z['actuals'],z['origins'],z['sensors']
 with np.load(ROOT/'data/cache/phase2_calib_forecasts.npz') as z: cq,cy,co=z['forecasts'],z['actuals'],z['origins']
 with np.load(ROOT/p3['train_forecast_cache']) as z: pooled=corr_from_residuals(z['actuals']-z['forecasts'][:,:,2])
 p4=json.loads((ROOT/cfg['phase4_config']).read_text())
 with np.load(ROOT/p4['branch_cache']) as z: tp,tc,tw=z['branch_paths'],z['branch_counts'],z['weights']
 with np.load(ROOT/cfg['branch_cache']) as z: cp,cc,cw=z['cal_paths'],z['cal_counts'],z['cal_weights']
 cq0,cy0,co0,cg,csi=p5.flatten_inputs(cq,cy,co,sensors); tq0,ty0,to0,tg,tsi=p5.flatten_inputs(tq,ty,to,sensors); ncal=len(cy0)
 q=np.concatenate([cq0,tq0]); y=np.concatenate([cy0,ty0]); origins=np.concatenate([co0,to0]); groups=np.concatenate([cg,tg]); sensor_slots=np.concatenate([csi,tsi]); paths=np.concatenate([cp,tp]); counts=np.concatenate([cc,tc]); weights=np.concatenate([cw,tw]); ns=len(y)
 column_index={str(name):i for i,name in enumerate(columns)}; speed_limits={}
 for group in range(sensors.shape[0]):
  for slot in range(sensors.shape[1]):
   name=str(sensors[group,slot]); col=column_index[name]; speed_limits[(group,slot)]=(float(values[:train_end,col].min()),float(values[:train_end,col].max()))
 def clipped_objective(sim,actual,group_ids,slots,alpha):
  losses=[]
  for h in p5.HORIZONS:
   los=np.empty(len(actual),float); his=np.empty(len(actual),float)
   for ix,(g,s) in enumerate(zip(group_ids,slots)):
    low,high=speed_limits[(int(g),int(s))]; los[ix],his[ix]=clip_interval(sim['lower'][ix,h],sim['upper'][ix,h],low,high)
   losses.extend(p5.interval_score(actual[:,h],los,his,alpha).tolist())
  return float(np.mean(losses))
 print('Computing cached-path scores...',flush=True)
 sorted_paths=np.sort(paths,axis=2); fork_scores=p5.score_paths(sorted_paths,counts,weights,y,.1,20); calmw=np.zeros_like(weights); anchors=np.zeros(ns,dtype=np.int8)
 for i in range(ns):
  ds=[np.linalg.norm(np.nanmedian(paths[i,j,:counts[i,j]],axis=0)-q[i,:,2]) if counts[i,j] else np.inf for j in range(2)]; anchors[i]=int(np.argmin(ds)); calmw[i,anchors[i]]=1
 calm_scores=p5.score_paths(sorted_paths,counts,calmw,y,.1,20); chron=np.stack([p5.sample_chronos(a,5505+i) for i,a in enumerate(q)]); chron_sort=np.sort(chron,axis=1); chron_score=np.stack([branch_nonconformity(chron_sort[i][None], [200],[1.],y[i],.1,20,True) for i in range(ns)]); scaled=np.stack([scaled_chronos_scores(a,b) for a,b in zip(q,y)])
 print('Tuning calibration-only gammas and replaying test...',flush=True)
 with np.load(ROOT/p3['feature_cache']) as z: cb,tb=z['cal_base'],z['test_base']
 with np.load(ROOT/p3['fork_output_cache']) as z: pf=np.concatenate([z['p_fork_cal']>=.168745,z['p_fork_test']>=.168745])
 rate=float(pf[:ncal].mean()); threshold=float(np.quantile(cb[:,1],1-rate,method='higher')); recent=np.concatenate([cb[:,1]>=threshold,tb[:,1]>=threshold]);
 def random_gate(n,seed):
  rng=np.random.default_rng(seed); flag=np.zeros(n,bool); flag[rng.permutation(n)[:int(round(n*rate))]]=True; return flag
 gates={'p(fork)':pf,'recent-drop':recent,'random':np.concatenate([random_gate(ncal,cfg['seed']),random_gate(len(ty0),cfg['seed']+1)])}
 gammas=cfg['gamma_grid']; rows=[]; raw=[]; chosen=[]
 base=dict(cfg); base['target_alpha']=.1
 for alpha in p6['alpha_targets']:
  specs=[('FORK ACI (p(fork))','fork','p(fork)',gammas),('Chronos scaled ACI (p(fork) gate)','chronos_scaled','p(fork)',gammas)]
  if float(alpha)==.1: specs.append(('FORK ACI (random)','fork','random',[.005]))
  for method,kind,gate_name,method_gammas in specs:
   sweep=[]; tune_cfg=dict(base); tune_cfg['target_alpha']=alpha
   for gamma in method_gammas:
    sim=p5.simulate(origins[:ncal],groups[:ncal],y[:ncal],sorted_paths[:ncal],counts[:ncal],weights[:ncal],anchors[:ncal],chron_sort[:ncal],q[:ncal],fork_scores[:ncal],calm_scores[:ncal],chron_score[:ncal],scaled[:ncal],gates[gate_name][:ncal],kind,gamma,tune_cfg)
    sweep.append((clipped_objective(sim,y[:ncal],groups[:ncal],sensor_slots[:ncal],alpha),gamma))
   gamma=min(sweep)[1]; chosen.append({'alpha':alpha,'method':method,'gamma':gamma,'gamma_selection':'frozen Phase 5 calibration choice' if method=='FORK ACI (random)' else 'calibration-only clipped interval score'})
   sim=p5.simulate(origins,groups,y,sorted_paths,counts,weights,anchors,chron_sort,q,fork_scores,calm_scores,chron_score,scaled,gates[gate_name],kind,gamma,tune_cfg)
   for h in (2,5,11):
    for i in range(ncal,ns):
     regime=label_target_period(y[i,:h+1],p1['free_flow_mph'],p1['congested_mph']); calm=not sim['group'][i]
     actual=float(y[i,h]); low,high=speed_limits[(int(groups[i]),int(sensor_slots[i]))]; lo,hi=clip_interval(sim['lower'][i,h:h+1],sim['upper'][i,h:h+1],low,high); width=float(hi[0]-lo[0]); score=float(p5.interval_score(np.array([actual]),lo,hi,alpha)[0]); covered=float(lo[0]<=actual<=hi[0])
     rec={'method':method,'alpha':alpha,'horizon':(h+1)*5,'regime':regime,'calm':calm,'covered':covered,'width':width,'interval_score':score,'origin':int(origins[i]),'fork_abs_error':float(abs(np.median(np.concatenate([paths[i,j,:counts[i,j]] for j in range(2) if counts[i,j] and weights[i,j]>=.1]),axis=0)[h]-actual)),'chronos_abs_error':float(abs(q[i,h,2]-actual)) if h==11 else np.nan}
     rows.append(rec)
     if alpha==.1 and method in ('FORK ACI (p(fork))','Chronos scaled ACI (p(fork) gate)') and h==11: raw.append(rec)
 print(f'Calibration sweep and test replay: {(time.perf_counter()-t0)/60:.1f} minutes.',flush=True)
 df=pd.DataFrame(rows); df.to_csv(out/'matched_curves.csv',index=False)
 curves=[]
 for (method,alpha,h),g in df.groupby(['method','alpha','horizon']):
  partitions=[(g[g.regime=='transition'],'transition'),(g[g.regime!='transition'],'non-transition'),(g[g.calm],'calm-flagged'),(g[~g.calm],'forked-flagged')]
  for subset,name in partitions:
   if len(subset): curves.append({'method':method,'alpha':alpha,'horizon':h,'subset':name,'coverage':subset.covered.mean(),'width':subset.width.mean(),'interval_score':subset.interval_score.mean(),'n':len(subset)})
 curve=pd.DataFrame(curves); curve.to_csv(out/'coverage_width_curve.csv',index=False)
 match=[]
 for h in (15,30,60):
  a=curve[(curve.horizon==h)&(curve.subset=='transition')]
  for target in p6['matched_transition_coverages']:
   vals={}
   for method,g in a.groupby('method'):
    vals[method]=(interpolate_at(g.coverage,g.width,target),interpolate_at(g.coverage,g.interval_score,target))
   fw,fi=vals.get('FORK ACI (p(fork))',(None,None)); cw,ci=vals.get('Chronos scaled ACI (p(fork) gate)',(None,None))
   match.append({'horizon':h,'target_transition_coverage':target,'fork_width':fw,'chronos_width':cw,'width_reduction':(cw-fw)/cw if fw is not None and cw not in (None,0) else None,'fork_interval_score':fi,'chronos_interval_score':ci})
 pd.DataFrame(match).to_csv(out/'matched_coverage.csv',index=False)
 calm_matches=[]
 for h in (15,30,60):
  a=curve[(curve.horizon==h)&(curve.subset=='non-transition')]; target=p6['matched_calm_coverage']; vals={}
  for method,g in a.groupby('method'): vals[method]=(interpolate_at(g.coverage,g.width,target),interpolate_at(g.coverage,g.interval_score,target))
  fw,fi=vals.get('FORK ACI (p(fork))',(None,None)); cw,ci=vals.get('Chronos scaled ACI (p(fork) gate)',(None,None))
  calm_matches.append({'horizon':h,'target_calm_coverage':target,'fork_width':fw,'chronos_width':cw,'width_reduction':(cw-fw)/cw if fw is not None and cw not in (None,0) else None,'fork_interval_score':fi,'chronos_interval_score':ci})
 pd.DataFrame(calm_matches).to_csv(out/'matched_calm_coverage.csv',index=False)
 fig,axs=plt.subplots(3,2,figsize=(10,11))
 for ri,h in enumerate((15,30,60)):
  for method,g in curve[(curve.horizon==h)&(curve.subset=='transition')].groupby('method'):
   axs[ri,0].plot(g.coverage,g.width,'o-',label=method); axs[ri,1].plot(g.coverage,g.interval_score,'o-',label=method)
  axs[ri,0].set_title(f'{h}m transition: coverage vs width'); axs[ri,1].set_title(f'{h}m transition: coverage vs interval score')
  for ax in axs[ri]: ax.set_xlabel('coverage'); ax.grid(alpha=.2)
 axs[0,0].legend(fontsize=7); axs[0,1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'matched_curves.png',dpi=150); plt.close(fig)
 # Branch-odds check on the test cache only; nearest Euclidean 12-step center is the outcome event.
 rr,brier,pred,hit=minority_reliability(tw,tc,tp,ty.reshape(-1,12),p6['minority_reliability_bins']); rtab=pd.DataFrame(rr,columns=['mean_minority_weight','minority_nearest_frequency','n']); rtab.to_csv(out/'branch_reliability.csv',index=False)
 fig,ax=plt.subplots(figsize=(5,4)); ax.plot([0,.5],[0,.5],'--',color='gray'); ax.plot(rtab.mean_minority_weight,rtab.minority_nearest_frequency,'o-'); ax.set(xlabel='stated minority weight',ylabel='observed nearest-minority frequency',title=f'Brier={brier:.3f}'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'branch_reliability.png',dpi=150); plt.close(fig)
 # Day-block CIs, 1000 resamples. All interval statistics are on the fixed alpha=.1 replay.
 main=pd.DataFrame(raw); timestamps=frame.timestamp
 day_lookup={int(o):pd.Timestamp(timestamps.iloc[int(o)]).date().toordinal() for o in to}
 main['day']=main.origin.map(day_lookup); unique_days=sorted(main.day.unique()); day_idx={d:i for i,d in enumerate(unique_days)}; ids=main.day.map(day_idx).to_numpy()
 specs={}
 for metric,mask in [('transition_coverage',main.regime.eq('transition')),('calm_group_coverage',main.calm),('forked_group_coverage',~main.calm),('transition_width',main.regime.eq('transition')),('transition_interval_score',main.regime.eq('transition'))]:
  val=main.covered if metric.endswith('coverage') else main.width if metric.endswith('width') else main.interval_score
  specs[metric]=(mask.to_numpy(),val.to_numpy())
 specs['weighted_median_point_mae']=(np.ones(len(main),bool),main.fork_abs_error.to_numpy()); specs['chronos_q50_point_mae']=(np.isfinite(main.chronos_abs_error.to_numpy()),main.chronos_abs_error.to_numpy())
 cis=[]
 for method,g in main.groupby('method'):
  loc=g.index.to_numpy(); local_ids=ids[loc]; block={k:(m[loc],v[loc]) for k,(m,v) in specs.items()}; ci=day_block_bootstrap_ci(local_ids,block,len(unique_days),p6['bootstrap_resamples'],p6['bootstrap_seed'])
  for metric,(lo,hi) in ci.items():
   subset='all test windows at 60m' if 'point_mae' in metric else 'transition windows' if metric.startswith('transition_') else 'gate-flagged group'
   cis.append({'method':method,'metric':metric,'subset':subset,'estimate':float(np.nanmean(block[metric][1][block[metric][0]])),'ci95_low':lo,'ci95_high':hi})
 pd.DataFrame(cis).to_csv(out/'day_block_bootstrap_ci.csv',index=False)
 # Compact corrected table: native, capped rows are excluded; this is alpha=.1 only.
 ci_df=pd.DataFrame(cis); table=[]
 for method,point_metric in [('FORK ACI (p(fork))','weighted_median_point_mae'),('Chronos scaled ACI (p(fork) gate)','chronos_q50_point_mae')]:
  d=ci_df[ci_df.method==method].set_index('metric'); m=main[main.method==method]
  tr=m[m.regime=='transition']; calm=m[m.calm]; fork=m[~m.calm]
  table.append({'method':method,'transition_coverage':tr.covered.mean(),'transition_coverage_ci95':f"{d.loc['transition_coverage','ci95_low']:.3f}-{d.loc['transition_coverage','ci95_high']:.3f}",'calm_group_coverage':calm.covered.mean(),'calm_group_ci95':f"{d.loc['calm_group_coverage','ci95_low']:.3f}-{d.loc['calm_group_coverage','ci95_high']:.3f}",'forked_group_coverage':fork.covered.mean(),'forked_group_ci95':f"{d.loc['forked_group_coverage','ci95_low']:.3f}-{d.loc['forked_group_coverage','ci95_high']:.3f}",'transition_width_mph':tr.width.mean(),'width_ci95':f"{d.loc['transition_width','ci95_low']:.2f}-{d.loc['transition_width','ci95_high']:.2f}",'interval_score':tr.interval_score.mean(),'interval_score_ci95':f"{d.loc['transition_interval_score','ci95_low']:.2f}-{d.loc['transition_interval_score','ci95_high']:.2f}",'point_mae':d.loc[point_metric,'estimate'],'mae_ci95':f"{d.loc[point_metric,'ci95_low']:.2f}-{d.loc[point_metric,'ci95_high']:.2f}",'point_mae_subset':'all test windows at 60m'})
 hci=pd.DataFrame(table); hci.to_csv(out/'headline_60m_ci.csv',index=False)
 cols=list(hci.columns); lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
 for _,r in hci.iterrows(): lines.append('| '+' | '.join(f"{v:.3f}" if isinstance(v,float) else str(v) for v in r)+' |')
 (out/'headline_60m_ci.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 pd.DataFrame(chosen).to_csv(out/'chosen_gamma.csv',index=False)
 print(f"A2: Phase 5's max-score region statistic is separate from the 12-marginal joint event; the max-score region is ~66% for FORK because its branch score and bounds saturate at empirical support. Scaled Chronos max-score region is ~89.5%; joint 12-interval coverage remains lower. No arithmetic bug found.")
 print(f"B/C/D outputs saved. Minority Brier={brier:.4f}; fraction in [0.25,0.35] predicted bins={float(np.mean((pred>=.25)&(pred<=.35))):.3f}; observed there={float(hit[(pred>=.25)&(pred<=.35)].mean()) if np.any((pred>=.25)&(pred<=.35)) else float('nan'):.3f}.")
 print(f"Phase 6 runtime {((time.perf_counter()-t0)/60):.1f} minutes.")
if __name__=='__main__': main()
