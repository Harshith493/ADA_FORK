"""Run the Phase 3 label-free fork detector evaluation."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,str(ROOT/'src'))

from fork.phase1 import (_load_period,_split_indices,_valid_test_origins,
    approximate_90_bounds,label_target_period,ChronosBoltCPU,QUANTILES,_sha256)
from fork.phase2 import forecast_contexts
from fork.window_matching import WindowMatcher
from fork.phase3 import (baseline_scores,binary_log_loss,calendar_features,
    detector_features,fit_logistic,generate_synthetic,kmeans_2,
    predict_logistic,precision_recall_curve,false_alarm_rate,threshold_by_f1)


def log(message, path):
    print(message,flush=True)
    with path.open('a',encoding='utf-8') as stream: stream.write(message+'\n')


def miss_targets(forecasts,actuals):
    low,high=approximate_90_bounds(forecasts)
    return ((actuals[...,11]<low[...,11])|(actuals[...,11]>high[...,11])).astype(int)


def valid_training_samples(values, timestamps, sensors, sensor_index, train_end, p1, count, seed):
    valid=_valid_test_origins(timestamps,p1['context_steps'],train_end,p1['context_steps'],12,p1['sampling_interval_minutes'])
    rng=np.random.default_rng(seed); candidates=[]
    for sensor in sorted(set(sensors.ravel().tolist())):
        # A 180-row stride makes each sensor's context plus target non-overlapping.
        offset=int(rng.integers(0,180))
        for origin in valid[offset::180]:
            candidates.append((sensor,int(origin)))
    if len(candidates)<count: raise ValueError(f'need {count} non-overlapping training windows; found {len(candidates)}')
    chosen=rng.choice(len(candidates),size=count,replace=False)
    selected=[candidates[i] for i in chosen]
    selected.sort(key=lambda item:(item[1],item[0]))
    contexts=np.stack([values[o-p1['context_steps']:o,sensor_index[s]] for s,o in selected])
    actuals=np.stack([values[o:o+12,sensor_index[s]] for s,o in selected])
    return selected,contexts.astype(np.float32),actuals.astype(np.float32),valid


def forecast_train_cache(cfg,p1,values,selected,contexts,actuals,data_hash,cache_path,log_path,force,other_estimate=0.):
    metadata={'format':'phase3-train-q-v1','sha256':data_hash,'model_id':p1['model_id'],
              'seed':cfg['train_sample_seed'],'count':len(selected),'context':p1['context_steps']}
    sensors=np.asarray([s for s,_ in selected]); origins=np.asarray([o for _,o in selected],int)
    if cache_path.exists() and not force:
        with np.load(cache_path,allow_pickle=False) as cache:
            old=json.loads(str(cache['metadata'].item()))
            if old==metadata and np.array_equal(cache['sensors'],sensors) and np.array_equal(cache['origins'],origins):
                return cache['forecasts'],cache['actuals'],'hit',0.
    model=ChronosBoltCPU(p1['model_id'],p1['torch_threads'])
    batch=cfg['forecast_batch_size']; pilot_n=min(batch,len(contexts))
    t0=time.perf_counter(); first=forecast_contexts(model,contexts[:pilot_n]); pilot=time.perf_counter()-t0
    projected=pilot*len(contexts)/pilot_n
    log(f'Train forecast pilot: {pilot_n} windows in {pilot:.2f}s; projected {projected/60:.1f} min for {len(contexts)} windows.',log_path)
    if projected+other_estimate>7200: raise RuntimeError('Combined forecast and matching estimate exceeds two hours; no full training forecast pass started.')
    q=np.empty((len(contexts),12,5),np.float32); q[:pilot_n]=first
    for start in range(pilot_n,len(contexts),batch):
        stop=min(start+batch,len(contexts))
        q[start:stop]=forecast_contexts(model,contexts[start:stop])
        if stop%1000<batch: log(f'Cached training forecasts: {stop}/{len(contexts)}',log_path)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,metadata=np.array(json.dumps(metadata,sort_keys=True)),
        forecasts=q,actuals=actuals,sensors=sensors,origins=origins)
    return q,actuals,'created',projected


def make_matchers(values,columns,timestamps,train_end,train_origins,sensors,p1):
    indices={s:i for i,s in enumerate(columns)}
    train_time=np.stack([calendar_features(pd.Timestamp(timestamps.iloc[o])) for o in train_origins])
    matchers={}
    for name in sorted(set(sensors.ravel().tolist())):
        matchers[name]=WindowMatcher(values[:train_end,indices[name]],p1['context_steps'],12,
            train_origins,candidate_time_features=train_time)
    return matchers,indices


def extract_set_features(items,values,columns,timestamps,matchers,indices,p1,train_end,k_grid,cfg,exclude_training=False):
    features={k:np.empty((len(items),4),float) for k in k_grid}
    base=np.empty((len(items),3),float)
    context_cache={}
    for i,(sensor,origin,q) in enumerate(items):
        key=(sensor,int(origin))
        if key not in context_cache:
            ctx=values[origin-p1['context_steps']:origin,indices[sensor]]
            tf=calendar_features(pd.Timestamp(timestamps.iloc[origin]))
            matcher=matchers[sensor]
            paths,neighbor_origins=matcher.query(ctx,max(k_grid),tf,
                exclude_origin=origin if exclude_training else None,
                exclusion_radius=cfg['train_exclusion_rows'] if exclude_training else 0)
            context_cache[key]=(ctx,paths,neighbor_origins,tf)
        ctx,paths,neighbor_origins,tf=context_cache[key]
        mean=float(values[:train_end,indices[sensor]].mean())
        std=max(float(values[:train_end,indices[sensor]].std()),1e-6)
        for k in k_grid:
            features[k][i]=detector_features(q,paths[:k],ctx,mean,std,
                seed=cfg['feature_seed']+i,n_init=cfg['kmeans_restarts'],
                max_iter=cfg['kmeans_iterations'])
        base[i]=baseline_scores(ctx,mean,std)
    return features,base


def score_rows(ytransition, scores, thresholds, cal_fraction, method_names):
    rows=[]; curves=[]
    for name,score in zip(method_names,scores):
        precision,recall,curve_thresholds,ap=precision_recall_curve(ytransition,score)
        flags=np.asarray(score)>=float(thresholds[name])
        rows.append({'method':name,'auprc':ap,'false_alarm_rate_calm':false_alarm_rate(ytransition,flags),
            'test_flag_rate':float(np.mean(flags)),'threshold':float(thresholds[name]),
            'calibration_miss_fraction':float(cal_fraction),'transition_n':int(np.sum(ytransition)),
            'calm_n':int(np.sum(~np.asarray(ytransition,dtype=bool)))})
        curves.extend({'method':name,'recall':float(r),'precision':float(p),'threshold':float(t)}
                      for p,r,t in zip(precision,recall,curve_thresholds))
    return pd.DataFrame(rows),pd.DataFrame(curves)


def main(force=False):
    cfg=json.loads((ROOT/'configs/phase3_pems_bay.json').read_text(encoding='utf-8'))
    p1=json.loads((ROOT/cfg['phase1_config']).read_text(encoding='utf-8'))
    out=ROOT/cfg['output_dir']; out.mkdir(parents=True,exist_ok=True)
    log_path=out/'run.log'; log_path.write_text('',encoding='utf-8')
    frame,columns=_load_period(ROOT/p1['dataset_path'],p1['period_start'],p1['period_end_exclusive'])
    values=frame[columns].to_numpy(np.float32); timestamps=frame['timestamp']
    train_end,cal_end=_split_indices(len(frame),p1['train_fraction'],p1['calibration_fraction'])
    data_hash=_sha256(ROOT/p1['dataset_path'])
    with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz',allow_pickle=False) as c:
        test_q,test_y,test_origins,sensors=c['forecasts'],c['actuals'],c['origins'],c['sensors']
    with np.load(ROOT/'data/cache/phase2_calib_forecasts.npz',allow_pickle=False) as c:
        cal_q,cal_y,cal_origins=c['forecasts'],c['actuals'],c['origins']
    selected,train_contexts,train_actuals,train_valid=valid_training_samples(
        values,timestamps,sensors,{s:i for i,s in enumerate(columns)},train_end,p1,
        cfg['train_window_count'],cfg['train_sample_seed'])
    train_origins=_valid_test_origins(timestamps,p1['context_steps'],train_end,p1['context_steps'],12,p1['sampling_interval_minutes'])
    matchers,indices=make_matchers(values,columns,timestamps,train_end,train_origins,sensors,p1)

    # Estimate the full inference and feature run before starting the full pass.
    pilot_items=selected[:min(20,len(selected))]
    t0=time.perf_counter()
    for i,(sensor,origin) in enumerate(pilot_items):
        ctx=values[origin-168:origin,indices[sensor]]
        tf=calendar_features(pd.Timestamp(timestamps.iloc[origin]))
        paths,_=matchers[sensor].query(ctx,max(cfg['k_grid']),tf,exclude_origin=origin,
            exclusion_radius=cfg['train_exclusion_rows'])
        mean=float(values[:train_end,indices[sensor]].mean()); std=float(values[:train_end,indices[sensor]].std())
        dummy_q=np.tile(np.array([1.,2.,3.,4.,5.]),(12,1))
        for k in cfg['k_grid']: detector_features(dummy_q,paths[:k],ctx,mean,std,
            cfg['feature_seed']+i,cfg['kmeans_restarts'],cfg['kmeans_iterations'])
    match_pilot=time.perf_counter()-t0
    all_feature_count=len(selected)+2*test_q.shape[0]*test_q.shape[1]*test_q.shape[2]
    match_est=match_pilot*all_feature_count/max(len(pilot_items),1)
    log(f'Feature pilot: {len(pilot_items)} train windows with kNN and 2-means in {match_pilot:.2f}s; projected {match_est/60:.1f} min for train/cal/test features.',log_path)
    train_cache=ROOT/cfg['train_forecast_cache']
    cache_exists=train_cache.exists() and not force
    forecast_estimate=0. if cache_exists else (18.*cfg['train_window_count']/4500.)
    log(f'Before full run: prior measured Chronos throughput estimates about {forecast_estimate/60:.1f} min for {cfg["train_window_count"]} training windows; combined estimate about {(forecast_estimate+match_est)/60:.1f} min.',log_path)
    if forecast_estimate+match_est>7200: raise RuntimeError('Combined projected run exceeds two hours; no full forecast pass started.')
    train_cache=ROOT/cfg['train_forecast_cache']
    train_q,train_actuals,train_status,train_est=forecast_train_cache(
        cfg,p1,values,selected,train_contexts,train_actuals,data_hash,train_cache,log_path,force,match_est)
    log(f'Training forecast cache={train_status}; non-overlap examples={len(selected)}; chronological blocks train={train_end}, cal={cal_end-train_end}, test={len(frame)-cal_end}.',log_path)

    # Cached feature arrays are keyed to model/data/config/sample identities.
    metadata={'format':'phase3-feature-v1','sha256':data_hash,'k_grid':cfg['k_grid'],
        'train_sensors':[s for s,_ in selected],'train_origins':[o for _,o in selected],
        'cal_origins':cal_origins.tolist(),'test_origins':test_origins.tolist(),
        'sensors':sensors.tolist(),'feature_seed':cfg['feature_seed']}
    fcache=ROOT/cfg['feature_cache']; feature_data=None
    if fcache.exists() and not force:
        with np.load(fcache,allow_pickle=False) as c:
            if json.loads(str(c['metadata'].item()))==metadata:
                feature_data={key:c[key] for key in c.files if key!='metadata'}
    if feature_data is None:
        train_items=[(s,o,q) for (s,o),q in zip(selected,train_q)]
        train_features,train_base=extract_set_features(train_items,values,columns,timestamps,matchers,indices,p1,train_end,cfg['k_grid'],cfg,True)
        cal_items=[]; test_items=[]
        for g in range(3):
            for si,s in enumerate(sensors[g]):
                for wi,o in enumerate(cal_origins): cal_items.append((s,int(o),cal_q[g,si,wi]))
                for wi,o in enumerate(test_origins): test_items.append((s,int(o),test_q[g,si,wi]))
        cal_features,cal_base=extract_set_features(cal_items,values,columns,timestamps,matchers,indices,p1,train_end,cfg['k_grid'],cfg,False)
        test_features,test_base=extract_set_features(test_items,values,columns,timestamps,matchers,indices,p1,train_end,cfg['k_grid'],cfg,False)
        feature_data={'train_base':train_base,'cal_base':cal_base,'test_base':test_base}
        for k in cfg['k_grid']:
            feature_data[f'train_f_{k}']=train_features[k]
            feature_data[f'cal_f_{k}']=cal_features[k]
            feature_data[f'test_f_{k}']=test_features[k]
        fcache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(fcache,metadata=np.array(json.dumps(metadata,sort_keys=True)),**feature_data)
        log('Feature cache created from train-only neighbors; calibration and test feature arrays contain no transition labels.',log_path)
    else:
        log('Feature cache=hit.',log_path)

    train_miss=miss_targets(train_q,train_actuals).reshape(-1)
    cal_miss=miss_targets(cal_q,cal_y).reshape(-1)
    # Select k and L2 only by calibration model-miss log loss.
    tuning=[]
    for k in cfg['k_grid']:
        xtr=feature_data[f'train_f_{k}']; xcal=feature_data[f'cal_f_{k}']
        for l2 in cfg['l2_grid']:
            model=fit_logistic(xtr,train_miss,l2=l2)
            pcal=predict_logistic(model,xcal)
            tuning.append({'k':k,'l2':l2,'calibration_miss_logloss':binary_log_loss(cal_miss,pcal)})
    tune_df=pd.DataFrame(tuning); tune_df.to_csv(out/'calibration_sweep.csv',index=False)
    selected_row=tune_df.sort_values(['calibration_miss_logloss','k','l2']).iloc[0]
    best_k=int(selected_row.k); best_l2=float(selected_row.l2)
    model=fit_logistic(feature_data[f'train_f_{best_k}'],train_miss,l2=best_l2)
    pcal=predict_logistic(model,feature_data[f'cal_f_{best_k}'])
    ptest=predict_logistic(model,feature_data[f'test_f_{best_k}'])
    cal_base=feature_data['cal_base']; test_base=feature_data['test_base']
    pcal_min=feature_data[f'cal_f_{best_k}'][:,1]; ptest_min=feature_data[f'test_f_{best_k}'][:,1]
    # Calibration-only hard-flag rate equals the approximate-90 calibration miss rate.
    miss_fraction=float(np.mean(cal_miss))
    p_threshold=float(np.quantile(pcal,1-miss_fraction,method='higher')) if miss_fraction>0 else float('inf')
    thresholds={'p(fork)':p_threshold}
    for j,name in enumerate(('volatility','recent-drop','CUSUM')):
        thresholds[name]=threshold_by_f1(cal_base[:,j],cal_miss)
    # Score transition labels only after all detector fit/tuning/threshold choices are frozen.
    test_trans=[]
    for g in range(3):
        for si in range(30):
            for wi in range(len(test_origins)):
                test_trans.append(int(label_target_period(test_y[g,si,wi,:12],p1['free_flow_mph'],p1['congested_mph'])=='transition'))
    test_trans=np.asarray(test_trans,int)
    score_mat=[ptest,test_base[:,0],test_base[:,1],test_base[:,2]]
    score_names=['p(fork)','volatility','recent-drop','CUSUM']
    metrics,curves=score_rows(test_trans,score_mat,thresholds,miss_fraction,score_names)
    metrics.to_csv(out/'detector_metrics.csv',index=False,float_format='%.6f')
    curves.to_csv(out/'precision_recall.csv',index=False,float_format='%.7f')
    fig,ax=plt.subplots(figsize=(7,6))
    for name in score_names:
        d=curves[curves.method==name]
        ax.plot(d.recall,d.precision,label=f'{name} (AP={metrics.loc[metrics.method==name,"auprc"].iloc[0]:.3f})')
    ax.set(xlabel='Recall for upcoming transition',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='PEMS-BAY test precision-recall (60 minutes)')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=170); plt.close(fig)

    # Cache p(fork) and minority-share arrays for Phases 4-5, before attaching labels.
    cache_path=ROOT/cfg['fork_output_cache']; cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,metadata=np.array(json.dumps({'format':'phase3-fork-output-v1','selected_k':best_k,'l2':best_l2,'data_sha256':data_hash},sort_keys=True)),
        p_fork_cal=pcal,p_fork_test=ptest,minority_share_cal=pcal_min,minority_share_test=ptest_min,
        test_origins=test_origins,sensors=sensors)
    prediction_rows=[]
    cursor=0
    for g in range(3):
        for si,s in enumerate(sensors[g]):
            for wi,o in enumerate(test_origins):
                prediction_rows.append({'seed':p1['sensor_seeds'][g],'sensor':s,'origin':int(o),
                    'p_fork':ptest[cursor],'minority_share':ptest_min[cursor],
                    'volatility':test_base[cursor,0],'recent_drop':test_base[cursor,1],
                    'cusum':test_base[cursor,2],'upcoming_transition_score_only':test_trans[cursor]})
                cursor+=1
    pred_df=pd.DataFrame(prediction_rows); pred_df.to_csv(out/'test_predictions.csv',index=False,float_format='%.7f')

    # Plot a calibration window chosen only by its detector feature, with branch share.
    cal_feats=feature_data[f'cal_f_{best_k}']; ex_idx=int(np.argmax(cal_feats[:,0]))
    cal_items=[]
    for g in range(3):
        for si,s in enumerate(sensors[g]):
            for wi,o in enumerate(cal_origins): cal_items.append((s,int(o),cal_q[g,si,wi]))
    es,eo,eq=cal_items[ex_idx]; ctx=values[eo-168:eo,indices[es]]; tf=calendar_features(pd.Timestamp(timestamps.iloc[eo]))
    neighbor_paths,_=matchers[es].query(ctx,best_k,tf)
    assignment,centers,_,share,_=kmeans_2((neighbor_paths-matchers[es].mean)/matchers[es].std,
        seed=cfg['feature_seed']+ex_idx,n_init=cfg['kmeans_restarts'],max_iter=cfg['kmeans_iterations'])
    fig,ax=plt.subplots(figsize=(8,5)); steps=np.arange(1,13)
    for cl,color in ((0,'#4c78a8'),(1,'#f58518')):
        for path in neighbor_paths[assignment==cl]:
            ax.plot(steps,path,alpha=.22,color=color,lw=.8)
        if np.any(assignment==cl): ax.plot(steps,neighbor_paths[assignment==cl].mean(axis=0),color=color,lw=2.5,label=f'branch {cl}, share={np.mean(assignment==cl):.2f}')
    ax.set(xlabel='5-minute step ahead',ylabel='Speed (mph)',title='Calibration neighbors split into two 60-minute futures')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'example_neighbor_branches.png',dpi=170); plt.close(fig)

    # Synthetic transfer check: detector remains PEMS-trained; fork outcomes are scoring-only.
    synthetic=generate_synthetic(cfg['synthetic_count'],p1['context_steps'],12,cfg['synthetic_seed'],
        cfg['synthetic_jump_mph'],cfg['synthetic_branch_coupling'])
    flat=synthetic['contexts'].reshape(-1,p1['context_steps'])
    model_forecaster=ChronosBoltCPU(p1['model_id'],p1['torch_threads'])
    syn_q=[]
    for start in range(0,len(flat),cfg['forecast_batch_size']):
        syn_q.append(forecast_contexts(model_forecaster,flat[start:start+cfg['forecast_batch_size']]))
    syn_q=np.concatenate(syn_q,axis=0).reshape(cfg['synthetic_count'],2,12,5)
    syn_scores=[]
    syn_min=[]
    for i in range(cfg['synthetic_count']):
        timestamp=pd.Timestamp('2020-01-01')+pd.Timedelta(minutes=5*i)
        tf=calendar_features(timestamp)
        for sidx in range(2):
            sensor=sorted(set(sensors.ravel().tolist()))[sidx]
            ctx=synthetic['contexts'][i,sidx]
            paths,_=matchers[sensor].query(ctx,best_k,tf)
            mean=float(values[:train_end,indices[sensor]].mean()); std=float(values[:train_end,indices[sensor]].std())
            x=detector_features(syn_q[i,sidx],paths,ctx,mean,std,seed=cfg['feature_seed']+i*2+sidx)
            syn_scores.append(predict_logistic(model,x[None,:])[0]); syn_min.append(x[1])
    syn_scores=np.asarray(syn_scores); syn_odds=np.repeat(synthetic['fork_odds'],2)
    syn_outcomes=synthetic['fork_outcomes'].reshape(-1)
    syn_ap=precision_recall_curve(syn_outcomes,syn_scores)[3]
    odds_rows=[]
    for pval in sorted(np.unique(syn_odds)):
        mask=syn_odds==pval
        odds_rows.append({'known_fork_odds':pval,'n_sensor_windows':int(mask.sum()),
            'mean_predicted_p_fork':float(syn_scores[mask].mean()),
            'realized_fork_rate':float(syn_outcomes[mask].mean())})
    odds_df=pd.DataFrame(odds_rows); odds_df.to_csv(out/'synthetic_odds_comparison.csv',index=False,float_format='%.6f')
    synthetic_metrics=pd.DataFrame([{'episodes':cfg['synthetic_count'],'sensor_windows':len(syn_scores),
        'known_mean_fork_odds':float(syn_odds.mean()),'predicted_mean_p_fork':float(syn_scores.mean()),
        'odds_mae_by_bin':float(np.mean(abs(odds_df.mean_predicted_p_fork-odds_df.known_fork_odds))),
        'brier_vs_known_odds':float(np.mean((syn_scores-syn_odds)**2)),
        'auprc_vs_realized_fork':syn_ap,'realized_fork_rate':float(syn_outcomes.mean()),
        'jump_mph':synthetic['jump_mph'],'coupling':synthetic['coupling']}])
    synthetic_metrics.to_csv(out/'synthetic_metrics.csv',index=False,float_format='%.6f')
    fig,ax=plt.subplots(figsize=(6,5))
    ax.plot(odds_df.known_fork_odds,odds_df.mean_predicted_p_fork,'o-',label='PEMS-trained detector')
    ax.plot([0,1],[0,1],ls='--',color='gray',label='identity')
    ax.set(xlabel='Generator-known fork odds',ylabel='Mean predicted p(fork)',title='Synthetic odds transfer check',xlim=(0,1),ylim=(0,1))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'synthetic_odds.png',dpi=170); plt.close(fig)

    best_baseline=metrics.loc[metrics.method.isin(score_names[1:])].sort_values('auprc',ascending=False).iloc[0]
    p_row=metrics.loc[metrics.method=='p(fork)'].iloc[0]
    volatility_ap=float(metrics.loc[metrics.method=='volatility','auprc'].iloc[0])
    kill=bool(p_row.auprc<=volatility_ap)
    log(f'Selected k={best_k}; L2={best_l2:g}; calibration miss fraction={miss_fraction:.3f}; hard-flag threshold={p_threshold:.5f}; test flag rate={p_row.test_flag_rate:.3f}.',log_path)
    log(f'Test AUPRC: p(fork)={p_row.auprc:.4f}; best baseline={best_baseline.method} {best_baseline.auprc:.4f}; calm false-alarm rate p(fork)={p_row.false_alarm_rate_calm:.3f}; kill/pivot={kill}.',log_path)
    log(f'Synthetic check: p mean={syn_scores.mean():.3f}, known odds mean={syn_odds.mean():.3f}, odds-bin MAE={synthetic_metrics.odds_mae_by_bin.iloc[0]:.3f}, AP={syn_ap:.3f}.',log_path)
    report=[f'# Phase 3: label-free fork detector','',
      '- Detector and tuning code receive no transition labels. The logistic target is the training/calibration approximate-90 Chronos miss indicator; transition labels are created only for final test scoring. PEMS-BAY is historical (2017) and may overlap Chronos training data.',
      f'- Train/calibration/test row counts: {train_end:,}/{cal_end-train_end:,}/{len(frame)-cal_end:,}; non-overlapping model training examples: {len(selected):,}; test score rows: {len(test_trans):,}.',
      f'- Forecast/training cache: {train_status}; test forecasts reused from Phase 1; calibration forecasts reused from Phase 2. Feature cache: {"hit" if fcache.exists() else "created"}.',
      f'- Selected k={best_k}, L2={best_l2:g} by calibration miss log loss. Hard flag is the calibration p(fork) quantile matching calibration miss fraction {miss_fraction:.1%}; threshold={p_threshold:.5f}; test flag rate={p_row.test_flag_rate:.1%}.',
      '', '## Detector results: 60-minute transition target, test only','',
      '| Score | AUPRC | False alarm rate on calm | Test flag rate | Threshold selection |','|---|---:|---:|---:|---|']
    for _,r in metrics.iterrows():
        selection='calibration miss-rate quantile' if r.method=='p(fork)' else 'calibration F1 vs miss'
        report.append(f"| {r.method} | {r.auprc:.3f} | {r.false_alarm_rate_calm:.3f} | {r.test_flag_rate:.3f} | {selection} |")
    report += ['',f"Best baseline by test AUPRC is {best_baseline.method} ({best_baseline.auprc:.3f}); p(fork) is {p_row.auprc:.3f}. Kill/pivot signal (p(fork) no better than volatility baseline): {'YES' if p_row.auprc<=metrics.loc[metrics.method=='volatility','auprc'].iloc[0] else 'NO'}.",
      'This signal means the fork score did not improve on volatility for transition ranking; do not claim that the detector is validated as a transition detector.',
      '', 'The output p(fork) is trained against Chronos interval misses as specified, so it is a fork-risk proxy, not a directly observed probability that the future has two latent branches. The synthetic known-odds check tests calibration transfer but does not redefine the supervised target.',
      '', '## Synthetic regime-switching check','',
      f"Two speed regimes, jump={synthetic['jump_mph']:.1f} mph, branch coupling={synthetic['coupling']:.2f}, {cfg['synthetic_count']} episodes/{len(syn_scores)} sensor windows. Known mean fork odds={syn_odds.mean():.3f}; mean p(fork)={syn_scores.mean():.3f}; odds-bin MAE={synthetic_metrics.odds_mae_by_bin.iloc[0]:.3f}; Brier vs odds={synthetic_metrics.brier_vs_known_odds.iloc[0]:.3f}; AP vs realized fork events={syn_ap:.3f}.",
      '', '## Reproducibility and leakage','',
      f"- Seeded training-window sample {cfg['train_sample_seed']}; feature seed {cfg['feature_seed']}; synthetic seed {cfg['synthetic_seed']}. Training windows are sampled on a 180-row stride per sensor, then seeded subsampled.",
      '- Historical neighbor search is training-only. Self-neighborhood exclusion is ±180 rows. Context normalization and calendar features use no test outcomes.',
      '- Four cyclical calendar coordinates are scaled by sqrt(168/4), balancing their combined distance contribution against the 168 normalized readings.',
      '- k/L2 and baseline thresholds use calibration miss targets only. Test transition labels are used only to score the frozen detector/baseline scores and false alarms.',
      '- Cached p(fork) and minority shares for Phase 4/5: data/cache/phase3_fork_outputs.npz. Detailed table/PR points and plot data are saved in outputs/phase3/.',
      '- Synthetic detector scores use the PEMS-trained model; generator fork outcomes and known odds are held back until scoring. It is a transfer stress test, not a claim of synthetic probability calibration.',
      '', 'No new packages or external code were used.']
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    log(f'Phase 3 complete. Outputs: {out}; fork cache: {ROOT/cfg["fork_output_cache"]}.',log_path)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true')
    main(ap.parse_args().force)
