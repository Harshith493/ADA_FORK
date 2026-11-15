"""Run Phase 5 prequential split/path conformal evaluation on cached forecasts."""
from __future__ import annotations
import argparse, hashlib, heapq, json, sys, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from fork.phase1 import _load_period,_split_indices,label_target_period,QUANTILES,approximate_90_bounds
from fork.phase2 import interval_score
from fork.phase3 import calendar_features
from fork.phase4 import corr_from_residuals,sample_branch_paths
from fork.phase5 import (ConformalState,MaturityQueue,branch_nonconformity,
    branch_bounds,central_bounds_at,scaled_chronos_scores,scaled_chronos_bounds)
from fork.window_matching import WindowMatcher

HORIZONS=(2,5,11); STEPS=(3,6,12); STREAMS=13

def make_calibration_branches(values,columns,timestamps,train_end,sensors,origins,qs,p1,p3,seed,pooled):
    context=p1['context_steps']; train_origins=np.arange(context,train_end-12+1)
    train_times=np.stack([calendar_features(pd.Timestamp(timestamps.iloc[int(o)])) for o in train_origins])
    indices={s:i for i,s in enumerate(columns)}; matchers={}
    for s in sorted(set(sensors.ravel().tolist())):
        matchers[s]=WindowMatcher(values[:train_end,indices[s]],context,12,train_origins,candidate_time_features=train_times)
    n=3*len(sensors[0])*len(origins); paths=np.full((n,2,200,12),np.nan,np.float32); counts=np.zeros((n,2),np.int16); weights=np.zeros((n,2),np.float32)
    idx=0
    for g in range(3):
      for si,s in enumerate(sensors[g]):
       matcher=matchers[s]
       for wi,o in enumerate(origins):
        context_values=values[int(o)-context:int(o),indices[s]]
        calendar=calendar_features(pd.Timestamp(timestamps.iloc[int(o)]))
        neighbors,_=matcher.query(context_values,25,calendar,exclude_origin=int(o),exclusion_radius=180)
        branch=__import__('fork.phase4',fromlist=['cluster_branches']).cluster_branches(neighbors,qs[g,si,wi,:,2],matcher.mean,matcher.std,seed=seed+wi+si+50000)
        _,parts,w,c=sample_branch_paths(branch,qs[g,si,wi],pooled,1.,.1,.5,200,seed+wi+g*100+si+50000,True)
        weights[idx]=w
        for j,p in parts.items(): counts[idx,j]=len(p); paths[idx,j,:len(p)]=p
        idx+=1
    return paths,counts,weights

def sample_chronos(q,seed):
    rng=np.random.default_rng(seed); u=rng.uniform(size=(200,q.shape[0]))
    return np.stack([np.interp(u[:,h],QUANTILES,q[h],left=q[h,0],right=q[h,-1]) for h in range(12)],axis=1).astype(np.float32)

def flatten_inputs(q,y,origins,sensors):
    return (q.reshape(-1,*q.shape[3:]),y.reshape(-1,*y.shape[3:]),
        np.tile(origins,3*len(sensors[0])),
        np.repeat(np.arange(3),len(sensors[0])*len(origins)),
        np.tile(np.repeat(np.arange(len(sensors[0])),len(origins)),3))

def score_paths(paths,counts,weights,actual,min_weight,steps):
    out=np.empty((len(actual),12),np.float32)
    for i in range(len(actual)):
        out[i]=branch_nonconformity(paths[i],counts[i],weights[i],actual[i],min_weight,steps,sorted_samples=True)
    return out

def finite_level(scores,alpha,target):
    if not scores: return 1.-target
    n=len(scores); rank=min(n,max(1,int(np.ceil((n+1)*(1-alpha)))))
    return float(np.partition(np.asarray(scores),rank-1)[rank-1])

def simulate(origins,groups,actual,branch_paths,branch_counts,branch_weights,anchors,chron_paths,quantiles,
             fork_scores,calm_scores,chron_scores,scaled_scores,gate,kind,gamma,cfg,collect=True):
    n=len(origins); order=np.lexsort((np.arange(n),origins)); state=ConformalState(STREAMS,cfg['target_alpha']); pending=MaturityQueue()
    tau_out=np.zeros((n,STREAMS),np.float32); alpha_out=np.zeros((n,STREAMS),np.float32); low=np.zeros((n,12),np.float32); high=np.zeros_like(low); path_low=np.zeros((n,12),np.float32); path_high=np.zeros_like(path_low); group_out=np.zeros(n,np.int8)
    def observe(i,stream,bounds):
        if kind=='chronos': scores=chron_scores
        elif kind=='chronos_scaled': scores=scaled_scores
        elif kind=='fork_collapse' and int(gate[i])==0: scores=calm_scores
        else: scores=fork_scores
        score=float(np.max(scores[i])) if stream==12 else float(scores[i,stream])
        if stream==12:
            y=actual[i]; lo_path,hi_path=bounds; miss=bool(np.any((y<lo_path)|(y>hi_path)))
        else:
            y=float(actual[i,stream]); lo_i,hi_i=bounds; miss=bool(y<lo_i or y>hi_i)
        return score,miss
    for i in order:
        o=int(origins[i]); pending.mature(o,state,gamma,cfg['target_alpha'],tuple(cfg['alpha_clip']),observe)
        g=int(gate[i]) if kind!='no_gate' else 0; group_out[i]=g
        collapse=(kind=='fork_collapse' and g==0)
        w=np.zeros(2,np.float32); w[int(anchors[i])]=1. if collapse else 0.
        use_weights=w if collapse else branch_weights[i]
        use_counts=branch_counts[i]
        paths=branch_paths[i]
        if kind in ('chronos','chronos_scaled'): paths0=chron_paths[i][None,:,:]; counts0=np.array([200]); weights0=np.array([1.]); paths0_sorted=True
        else: paths0=paths; counts0=use_counts; weights0=use_weights; paths0_sorted=True
        for stream in range(STREAMS):
            alpha_now=state._group(g)['alpha'][stream]; tau=state.level(g,stream,cfg['target_alpha'])
            tau_out[i,stream]=tau; alpha_out[i,stream]=alpha_now
            if stream<12:
                h=stream
                if kind=='chronos_scaled':
                    lo0,hi0=scaled_chronos_bounds(quantiles[i],tau); lo=float(lo0[h]); hi=float(hi0[h])
                else: lo,hi=central_bounds_at(paths0,counts0,weights0,h,tau,cfg['branch_drop_threshold'],paths0_sorted)
                low[i,h]=lo; high[i,h]=hi; maturity=o+h+1
            else:
                if kind=='chronos_scaled':
                    lo_path,hi_path=scaled_chronos_bounds(quantiles[i],tau)
                else: lo_path,hi_path=branch_bounds(paths0,counts0,weights0,tau,cfg['branch_drop_threshold'],paths0_sorted)
                path_low[i]=lo_path; path_high[i]=hi_path
                maturity=o+12
            pending.push(maturity,g,stream,reference=int(i),bounds=(lo_path,hi_path) if stream==12 else (lo,hi))
    path_covered=np.all((actual>=path_low)&(actual<=path_high),axis=1)
    joint_covered=np.all((actual>=low)&(actual<=high),axis=1)
    return {'tau':tau_out,'alpha':alpha_out,'lower':low,'upper':high,'path_covered':path_covered,'joint_covered':joint_covered,'group':group_out,'score':tau_out}

def objective(result,actual,alpha=.1):
    losses=[]
    for h in HORIZONS: losses.extend(interval_score(actual[:,h],result['lower'][:,h],result['upper'][:,h],alpha).tolist())
    return float(np.mean(losses))

def make_labels(actual,p1):
    return np.asarray([label_target_period(y,p1['free_flow_mph'],p1['congested_mph']) for y in actual],object)

def write_md(df,path):
    cols=list(df.columns)
    def c(x): return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','\\|')
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    lines.extend('| '+' | '.join(c(r[k]) for k in cols)+' |' for _,r in df.iterrows())
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/phase5_pems_bay.json'); args=ap.parse_args()
    cfg=json.loads((ROOT/args.config).read_text()); p1=json.loads((ROOT/cfg['phase1_config']).read_text()); p3=json.loads((ROOT/cfg['phase3_config']).read_text()); p4=json.loads((ROOT/cfg['phase4_config']).read_text())
    out=ROOT/cfg['output_dir']; out.mkdir(parents=True,exist_ok=True); log=out/'run.log'; log.write_text('',encoding='utf-8')
    frame,columns=_load_period(ROOT/p1['dataset_path'],p1['period_start'],p1['period_end_exclusive']); values=frame[columns].to_numpy(np.float32); timestamps=frame.timestamp
    train_end,cal_end=_split_indices(len(frame),p1['train_fraction'],p1['calibration_fraction'])
    with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz',allow_pickle=False) as z: tq,ty,to,sensors=z['forecasts'],z['actuals'],z['origins'],z['sensors']
    with np.load(ROOT/'data/cache/phase2_calib_forecasts.npz',allow_pickle=False) as z: cq,cy,co=z['forecasts'],z['actuals'],z['origins']
    with np.load(ROOT/p3['train_forecast_cache'],allow_pickle=False) as z: train_res=z['actuals']-z['forecasts'][:,:,2]
    pooled=corr_from_residuals(train_res); n=4500
    with np.load(ROOT/p4['branch_cache'],allow_pickle=False) as z: test_paths=z['branch_paths']; test_counts=z['branch_counts']; test_weights=z['weights']; p4meta=json.loads(str(z['metadata'].item()))
    if p4meta.get('alpha')!=1.0 or p4meta.get('threshold')!=.1 or p4meta.get('shrinkage')!=.5: raise ValueError('Phase 4 cache settings differ from frozen Phase 5 choices')
    if p4meta.get('origins')!=to.tolist() or p4meta.get('sensors')!=sensors.tolist(): raise ValueError('Phase 4 test cache does not match Phase 1 forecast selection')
    cache_path=ROOT/cfg['branch_cache']; data_hash=hashlib.sha256((ROOT/p1['dataset_path']).read_bytes()).hexdigest()
    meta={'format':'phase5-branches-v1','sha256':data_hash,'origins':co.tolist(),'sensors':sensors.tolist(),'frozen_p4':{'alpha':1.,'threshold':.1,'shrinkage':.5,'k':25,'paths':200}}
    cal_paths=cal_counts=cal_weights=None
    if cache_path.exists():
        with np.load(cache_path,allow_pickle=False) as z:
            if json.loads(str(z['metadata'].item()))==meta: cal_paths=z['cal_paths']; cal_counts=z['cal_counts']; cal_weights=z['cal_weights']
    if cal_paths is None:
        # Estimate matching + sampling before the calibration pass; this is not a Chronos run.
        start=time.perf_counter(); pilot=0
        for g in range(3):
          for si,s in enumerate(sensors[g][:1]):
            break
        # Build matchers once and time 20 representative windows.
        train_origins=np.arange(168,train_end-12+1); tf=np.stack([calendar_features(pd.Timestamp(timestamps.iloc[int(o)])) for o in train_origins]); si_map={s:i for i,s in enumerate(columns)}
        matchers={s:WindowMatcher(values[:train_end,si_map[s]],168,12,train_origins,candidate_time_features=tf) for s in sorted(set(sensors.ravel().tolist()))}
        t0=time.perf_counter(); seen=0
        for g in range(3):
          for si,s in enumerate(sensors[g]):
            for wi,o in enumerate(co[:1]):
                m=matchers[s]; hist,_=m.query(values[int(o)-168:int(o),si_map[s]],25,calendar_features(pd.Timestamp(timestamps.iloc[int(o)])),exclude_origin=int(o),exclusion_radius=180)
                br=__import__('fork.phase4',fromlist=['cluster_branches']).cluster_branches(hist,cq[g,si,wi,:,2],m.mean,m.std,seed=cfg['seed']+wi+si)
                sample_branch_paths(br,cq[g,si,wi],pooled,1.,.1,.5,200,cfg['seed']+wi+si,True); seen+=1
                if seen>=20: break
            if seen>=20: break
          if seen>=20: break
        pilot=time.perf_counter()-t0; estimate=pilot*n/seen
        print(f'Branch-generation pilot: {seen} windows in {pilot:.2f}s; projected {estimate/60:.1f} min for {n} calibration windows.',flush=True)
        if estimate>7200: raise RuntimeError('Calibration branch-generation estimate exceeds two hours; did not start full pass.')
        t0=time.perf_counter(); cal_paths,cal_counts,cal_weights=make_calibration_branches(values,columns,timestamps,train_end,sensors,co,cq,p1,p3,cfg['seed'],pooled)
        np.savez_compressed(cache_path,metadata=np.array(json.dumps(meta,sort_keys=True)),cal_paths=cal_paths,cal_counts=cal_counts,cal_weights=cal_weights)
        print(f'Calibration branches generated in {(time.perf_counter()-t0)/60:.1f} min and cached.',flush=True)
    else: print('Calibration branch cache hit.',flush=True)
    # Flatten using the same seed/sensor/origin order as Phase 1-4 caches.
    cal_q,cal_y,cal_o,cal_g,cal_si=flatten_inputs(cq,cy,co,sensors); test_q,test_y,test_o,test_g,test_si=flatten_inputs(tq,ty,to,sensors)
    fork_paths=np.concatenate([cal_paths,test_paths]); fork_counts=np.concatenate([cal_counts,test_counts]); fork_weights=np.concatenate([cal_weights,test_weights])
    qs=np.concatenate([cal_q,test_q]); ys=np.concatenate([cal_y,test_y]); origins=np.concatenate([cal_o,test_o]); seedgroups=np.concatenate([cal_g,test_g]); sensor_ids=np.concatenate([cal_si,test_si]); ncal=len(cal_y)
    # Score arrays use sorted sample values for fast, reproducible interval inversion.
    print('Computing calibration/test nonconformity scores...',flush=True); t0=time.perf_counter()
    fork_sorted=np.sort(fork_paths,axis=2); fork_scores=score_paths(fork_sorted,fork_counts,fork_weights,ys,cfg['branch_drop_threshold'],cfg['score_bisection_steps'])
    anchor=np.zeros(len(ys),np.int8); calm_weights=np.zeros_like(fork_weights)
    for i in range(len(ys)):
        dist=[]
        for j in range(2):
            if fork_counts[i,j]>0: dist.append(np.linalg.norm(np.nanmedian(fork_paths[i,j,:fork_counts[i,j]],axis=0)-qs[i,:,2]))
            else: dist.append(np.inf)
        anchor[i]=int(np.argmin(dist)); calm_weights[i,anchor[i]]=1.
    calm_scores=score_paths(fork_sorted,fork_counts,calm_weights,ys,cfg['branch_drop_threshold'],cfg['score_bisection_steps'])
    chron_paths=np.stack([sample_chronos(q,cfg['seed']+i) for i,q in enumerate(qs)]); chron_sorted=np.sort(chron_paths,axis=1)
    chron_scores=np.stack([branch_nonconformity(chron_sorted[i][None,:,:],[200],[1.],ys[i],cfg['branch_drop_threshold'],cfg['score_bisection_steps'],True) for i in range(len(ys))])
    scaled_scores=np.stack([scaled_chronos_scores(q,y) for q,y in zip(qs,ys)]).astype(np.float32)
    print(f'Score arrays ready in {(time.perf_counter()-t0)/60:.1f} min.',flush=True)
    # Gate choices are label-free apart from the isolated oracle reference.
    with np.load(ROOT/p3['feature_cache'],allow_pickle=False) as z: cal_base=z['cal_base']; test_base=z['test_base']
    with np.load(ROOT/p3['fork_output_cache'],allow_pickle=False) as z: pcal=z['p_fork_cal']; ptest=z['p_fork_test']
    pflags=np.concatenate([pcal>=.168745,ptest>=.168745]); cal_rate=float(np.mean(pcal>=.168745))
    recent_values=cal_base[:,1]; recent_thr=float(np.quantile(recent_values,1-cal_rate,method='higher'))
    recent_flags=np.concatenate([cal_base[:,1]>=recent_thr,test_base[:,1]>=recent_thr])
    branch_flags=np.all((fork_counts>0)&(fork_weights>0),axis=1)
    def exact_random(n,rate,seed):
        rng=np.random.default_rng(seed); result=np.zeros(n,bool); result[rng.permutation(n)[:int(round(n*rate))]]=True; return result
    random_flags=np.concatenate([exact_random(ncal,cal_rate,cfg['seed']),exact_random(len(test_y),cal_rate,cfg['seed']+1)])
    labels=make_labels(ys,p1); oracle_flags=(labels=='transition')
    gates={'p(fork)':pflags,'recent-drop':recent_flags,'branch-weight':branch_flags,'random':random_flags,'no-gate':np.zeros(len(ys),bool),'oracle (reference only)':oracle_flags}
    pd.DataFrame([{'gate':name,'calibration_flag_rate':float(np.mean(flag[:ncal])),'test_flag_rate':float(np.mean(flag[ncal:])),'threshold':(.168745 if name=='p(fork)' else recent_thr if name=='recent-drop' else np.nan),'oracle_uses_labels':name=='oracle (reference only)'} for name,flag in gates.items()]).to_csv(out/'gate_summary.csv',index=False,float_format='%.6f')
    specs=[('FORK ACI (p(fork))','fork','p(fork)'),('FORK ACI (recent-drop)','fork','recent-drop'),('FORK ACI (branch-weight)','fork','branch-weight'),('FORK ACI (random)','fork','random'),('FORK ACI (no gate)','no_gate','no-gate'),('FORK ACI (oracle reference)','fork','oracle (reference only)'),('FORK ACI (p(fork), calm collapse)','fork_collapse','p(fork)'),('Chronos ACI (capped, not a valid baseline; no gate)','chronos','no-gate'),('Chronos ACI (capped, not a valid baseline; p(fork))','chronos','p(fork)'),('Chronos scaled ACI (no gate)','chronos_scaled','no-gate'),('Chronos scaled ACI (p(fork) gate)','chronos_scaled','p(fork)')]
    choices={}; sweep=[]
    for name,kind,gate_name in specs:
        if 'oracle' in name: continue
        for gamma in cfg['gamma_grid']:
            sim=simulate(origins[:ncal],seedgroups[:ncal],ys[:ncal],fork_sorted[:ncal],fork_counts[:ncal],fork_weights[:ncal],anchor[:ncal],chron_sorted[:ncal],cal_q,fork_scores[:ncal],calm_scores[:ncal],chron_scores[:ncal],scaled_scores[:ncal],gates[gate_name][:ncal],kind,gamma,cfg)
            val=objective(sim,ys[:ncal]); sweep.append({'method':name,'gamma':gamma,'calibration_interval_score':val})
        best=min((r for r in sweep if r['method']==name),key=lambda r:(r['calibration_interval_score'],r['gamma']))
        choices[name]=float(best['gamma'])
    choices['FORK ACI (oracle reference)']=choices['FORK ACI (p(fork))']
    pd.DataFrame(sweep).to_csv(out/'gamma_sweep.csv',index=False)
    # Evaluate every method continuously over calibration then test; scores mature only at their target time.
    result_by_method={}; alpha_rows=[]
    for name,kind,gate_name in specs:
        gamma=choices[name]
        sim=simulate(origins,seedgroups,ys,fork_sorted,fork_counts,fork_weights,anchor,chron_sorted,qs,fork_scores,calm_scores,chron_scores,scaled_scores,gates[gate_name],kind,gamma,cfg)
        sim['gate']=gates[gate_name]; sim['method']=name; result_by_method[name]=sim
        for i in range(ncal,len(ys)):
            alpha_rows.append({'method':name,'origin':int(origins[i]),'flag_group':'forked' if sim['group'][i] else ('all' if gate_name=='no-gate' else 'calm'),**{f'alpha_{k}':float(sim['alpha'][i,k]) for k in range(STREAMS)}})
    # Labels are used only in this final, post-prediction scoring section.
    test_labels=labels[ncal:]; methods=[]; headline=[]; path_rows=[]; point_rows=[]
    for name,sim in result_by_method.items():
        for i in range(ncal,len(ys)):
            g=int(sim['group'][i]); grp='all' if 'no gate' in name.lower() else ('forked-flagged' if g else 'calm-flagged')
            for j,h in enumerate(HORIZONS):
                lab=label_target_period(ys[i,:h+1],p1['free_flow_mph'],p1['congested_mph'])
                methods.append({'method':name,'flag_group':grp,'regime':lab,'horizon_minutes':STEPS[j]*5,'actual':ys[i,h],'lower':sim['lower'][i,h],'upper':sim['upper'][i,h],'interval_score':interval_score(ys[i,h],sim['lower'][i,h],sim['upper'][i,h],.1),'origin':origins[i]})
            path_rows.append({'method':name,'flag_group':grp,'path_region_covered':sim['path_covered'][i],'joint_12_covered':sim['joint_covered'][i],'regime_60m':test_labels[i-ncal],'origin':origins[i]})
    raw_test=pd.DataFrame(methods)
    metrics=raw_test.groupby(['method','flag_group','regime','horizon_minutes']).apply(lambda x:pd.Series({'n':len(x),'coverage':np.mean((x.actual>=x.lower)&(x.actual<=x.upper)),'mean_width_mph':np.mean(x.upper-x.lower),'interval_score':x.interval_score.mean()}),include_groups=False).reset_index()
    metrics.to_csv(out/'results.csv',index=False,float_format='%.6f'); write_md(metrics,out/'results.md')
    pathdf=pd.DataFrame(path_rows)
    pathsum=pathdf.groupby(['method','flag_group']).agg(path_n=('path_region_covered','count'),whole_path_region_coverage=('path_region_covered','mean'),joint_12_interval_coverage=('joint_12_covered','mean')).reset_index()
    pathsum.to_csv(out/'whole_path_coverage.csv',index=False,float_format='%.6f')
    # Compact 60-minute table combines flagged-group coverage, true-regime coverage, widths and path coverage.
    headrows=[]
    for method in raw_test.method.unique():
        sub=raw_test[(raw_test.method==method)&(raw_test.horizon_minutes==60)]
        def cov(x): return float(np.mean((x.actual>=x.lower)&(x.actual<=x.upper))) if len(x) else np.nan
        def width(x): return float(np.mean(x.upper-x.lower)) if len(x) else np.nan
        def score(x): return float(x.interval_score.mean()) if len(x) else np.nan
        pc=pathsum[pathsum.method==method]
        def pcover(g):
            row=pc[pc.flag_group==g]
            return float(row.whole_path_region_coverage.iloc[0]) if len(row) else np.nan
        def jcover(g):
            row=pc[pc.flag_group==g]
            return float(row.joint_12_interval_coverage.iloc[0]) if len(row) else np.nan
        calm=sub[sub.flag_group=='calm-flagged']; forked=sub[sub.flag_group=='forked-flagged']; allrows=sub
        tr=sub[sub.regime=='transition']; ff=sub[sub.regime=='free-flow']; co2=sub[sub.regime=='congested']
        headrows.append({'method':method,'n':len(sub),'calm_flag_coverage':cov(calm),'forked_flag_coverage':cov(forked),'all_group_coverage':cov(allrows),
          'free_flow_coverage':cov(ff),'free_flow_width_mph':width(ff),'free_flow_interval_score':score(ff),
          'transition_coverage':cov(tr),'transition_width_mph':width(tr),'transition_interval_score':score(tr),
          'congested_coverage':cov(co2),'congested_width_mph':width(co2),'congested_interval_score':score(co2),'mean_width_mph':width(allrows),
          'calm_flag_path_region_coverage':pcover('calm-flagged'),'forked_flag_path_region_coverage':pcover('forked-flagged'),'all_path_region_coverage':pcover('all'),
          'calm_flag_joint_12_coverage':jcover('calm-flagged'),'forked_flag_joint_12_coverage':jcover('forked-flagged'),'all_joint_12_coverage':jcover('all')})
    headline=pd.DataFrame(headrows); headline.to_csv(out/'headline_60m.csv',index=False,float_format='%.6f'); write_md(headline,out/'headline_60m.md')
    pd.DataFrame(alpha_rows).to_csv(out/'alpha_trace.csv',index=False)
    # Point forecast comparison for the main FORK gate, weighted empirical median versus Chronos q50.
    main=result_by_method['FORK ACI (p(fork))']; main_test=ncal
    fork_point=[]; chron_point=[]; point_rows=[]
    for i in range(main_test,len(ys)):
        active=[j for j in range(2) if fork_counts[i,j]>0 and fork_weights[i,j]>=cfg['branch_drop_threshold']]
        if not active: active=[int(np.argmax(fork_weights[i]))]
        draw=np.concatenate([fork_paths[i,j,:fork_counts[i,j]] for j in active]); point=np.median(draw,axis=0)
        fork_point.extend(np.abs(point[[2,5,11]]-ys[i,[2,5,11]]).tolist())
        chron_point.extend(np.abs(qs[i,[2,5,11],2]-ys[i,[2,5,11]]).tolist())
        for h in HORIZONS:
            point_rows.append({'origin':int(origins[i]),'horizon_minutes':(h+1)*5,'fork_mae':float(abs(point[h]-ys[i,h])),'chronos_q50_mae':float(abs(qs[i,h,2]-ys[i,h]))})
    pd.DataFrame(point_rows).to_csv(out/'point_predictions.csv',index=False,float_format='%.6f')
    mae=pd.DataFrame([{'method':'FORK weighted branch median','point_mae':float(np.mean(fork_point)),'subset':'all test windows at 15/30/60m'},{'method':'Chronos q50','point_mae':float(np.mean(chron_point)),'subset':'all test windows at 15/30/60m'}]); mae.to_csv(out/'point_mae.csv',index=False,float_format='%.6f')
    headline['fork_weighted_median_mae']=np.nan; headline['chronos_q50_mae']=np.nan
    headline.loc[headline.method=='FORK ACI (p(fork))','fork_weighted_median_mae']=float(np.mean(fork_point))
    headline.loc[headline.method=='FORK ACI (p(fork))','chronos_q50_mae']=float(np.mean(chron_point))
    headline['point_mae_subset']='all test windows at 15/30/60m'
    headline.to_csv(out/'headline_60m.csv',index=False,float_format='%.6f'); write_md(headline,out/'headline_60m.md')
    # Coverage-width comparison: test-only descriptive check, never used for parameter selection.
    head=raw_test[(raw_test.horizon_minutes==60)&(raw_test.regime=='transition')]
    def lookup(method):
        x=head[head.method==method]
        if len(x)==0:return None
        return {'coverage':float(np.mean((x.actual>=x.lower)&(x.actual<=x.upper))),
                'width':float(np.mean(x.upper-x.lower))}
    fm=lookup('FORK ACI (p(fork))'); cm=lookup('Chronos scaled ACI (no gate)'); cg=lookup('Chronos scaled ACI (p(fork) gate)')
    verdict=[]
    if fm and cm and cg:
        f_cov=fm['coverage']; f_width=fm['width']; c_cov=[cm['coverage'],cg['coverage']]; c_width=[cm['width'],cg['width']]
        matched=[j for j,v in enumerate(c_cov) if abs(v-f_cov)<=cfg['matched_coverage_tolerance']]
        gain=max([(c_width[j]-f_width)/c_width[j] for j in matched],default=np.nan)
        verdict=[f"FORK transition coverage={f_cov:.1%}; target 85% {'reached' if f_cov>=.85 else 'not reached'}.",
                 f"Matched-coverage baselines within ±{cfg['matched_coverage_tolerance']:.0%}: {', '.join(['Chronos scaled ACI (no gate)' if j==0 else 'Chronos scaled ACI (p(fork) gate)' for j in matched]) or 'none'}.",
                 (f"FORK width reduction vs best matched baseline={gain:.1%} {'(≥10% sharper)' if gain>=.10 else '(below 10%)'}." if np.isfinite(gain) else "No matched-coverage verdict; a ≥10% width-reduction claim is unestablished.")]
    else: verdict=['D8(e) verdict unavailable: transition rows are missing.']
    (out/'D8e_verdict.md').write_text('\n'.join(verdict)+'\n',encoding='utf-8')
    # Plots.
    plot=metrics[(metrics.horizon_minutes==60)&(metrics.regime=='transition')]
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(plot.mean_width_mph,plot.coverage)
    for _,r in plot.iterrows(): ax.annotate(r.method,(r.mean_width_mph,r.coverage),fontsize=7)
    ax.axhline(.85,color='gray',ls='--'); ax.set(xlabel='Transition mean width (mph)',ylabel='Transition coverage (60m)',title='Phase 5 coverage vs width'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'coverage_vs_width.png',dpi=160); plt.close(fig)
    at=pd.DataFrame(alpha_rows)
    fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True)
    main_trace=at[at.method=='FORK ACI (p(fork))']
    for g,group in enumerate(('calm','forked')):
        sub=main_trace[main_trace.flag_group==f'{g}-flagged'].sort_values('origin')
        for j,label in zip((2,5,11,12),('15m','30m','60m','path')): axes[g,0].plot(np.arange(len(sub)),sub[f'alpha_{j}'],label=label)
        axes[g,0].set_title(f'{g} gate state'); axes[g,0].set_ylabel('alpha_t'); axes[g,0].grid(alpha=.2)
    axes[0,0].legend(ncol=2,fontsize=8); axes[1,0].set_xlabel('Test windows in saved stream order'); axes[0,1].axis('off'); axes[1,1].axis('off'); fig.tight_layout(); fig.savefig(out/'alpha_over_time.png',dpi=160); plt.close(fig)
    cum=raw_test[(raw_test.method=='FORK ACI (p(fork))')&(raw_test.horizon_minutes==60)]
    fig,ax=plt.subplots(figsize=(8,5))
    for group in ('calm-flagged','forked-flagged'):
        z=cum[cum.flag_group==group].sort_values(['origin'])
        hit=((z.actual>=z.lower)&(z.actual<=z.upper)).astype(float).to_numpy()
        ax.plot(np.arange(1,len(hit)+1),np.cumsum(hit)/np.arange(1,len(hit)+1),label=group)
    ax.axhline(.9,color='black',ls='--',lw=1); ax.set(xlabel='Test scored windows (origin order)',ylabel='Cumulative 60-minute coverage',title='Coverage convergence by p(fork) group'); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'cumulative_coverage.png',dpi=160); plt.close(fig)
    pd.DataFrame([{'method':k,'gamma':v} for k,v in choices.items()]).to_csv(out/'selected_gamma.csv',index=False)
    print(f'Phase 5 complete. Calibration branch cache: {cache_path}; selected gammas={choices}; outputs={out}',flush=True)

if __name__=='__main__': main()
