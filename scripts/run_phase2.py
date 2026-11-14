"""Run Phase 2 using cached Phase 1 test forecasts and a separate calibration cache."""
from __future__ import annotations
import argparse, json, sys, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0, str(ROOT / "src"))
from fork.phase1 import (_load_period, _split_indices, _valid_test_origins, select_window_origins,
                         ChronosBoltCPU, approximate_90_bounds, label_target_period, REGIMES, QUANTILES)
from fork.window_matching import WindowMatcher, empirical_interval
from fork.phase2 import interval_score, aci_update, widen_bounds, mix_samples, quantiles_to_samples, forecast_contexts

def main(force=False):
    cfg = json.loads((ROOT/'configs/phase2_pems_bay.json').read_text())
    p1 = json.loads((ROOT/cfg['phase1_config']).read_text())
    frame, cols = _load_period(ROOT/p1['dataset_path'], p1['period_start'], p1['period_end_exclusive'])
    vals = frame[cols].to_numpy(np.float32)
    tr_end, cal_end = _split_indices(len(frame), p1['train_fraction'], p1['calibration_fraction'])
    with np.load(ROOT/'data/cache/phase1_pems_bay_forecasts.npz', allow_pickle=False) as c:
        tf, ty, to, sensors = c['forecasts'], c['actuals'], c['origins'], c['sensors']
    cal_valid = _valid_test_origins(frame.timestamp, tr_end, cal_end, p1['context_steps'], 12, 5)
    co = select_window_origins(cal_valid, cfg['calibration_window_count'], cfg['calibration_seed'])
    ccpath = ROOT/cfg['calibration_cache']
    if ccpath.exists() and not force:
        with np.load(ccpath, allow_pickle=False) as c:
            if not np.array_equal(c['origins'], co) or not np.array_equal(c['sensors'], sensors):
                raise ValueError('calibration cache selection mismatch; rerun with --force')
            cf, cy = c['forecasts'], c['actuals']
        cache_status='hit'
    else:
        model=ChronosBoltCPU(p1['model_id'], p1['torch_threads'])
        cf=np.empty((3,30,len(co),12,5),np.float32); cy=np.empty((3,30,len(co),12),np.float32)
        ix={s:i for i,s in enumerate(cols)}
        for g,group in enumerate(sensors):
            si=[ix[s] for s in group]
            for wi,o in enumerate(co):
                cf[g,:,wi]=forecast_contexts(model,vals[o-p1['context_steps']:o,si].T)
                cy[g,:,wi]=vals[o:o+12,si].T
        ccpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(ccpath,forecasts=cf,actuals=cy,origins=co,sensors=sensors)
        cache_status='created'
    print(f'calibration cache={cache_status}; calibration origins={len(co)}; test cache reused={tf.shape};')

    # Cache nearest historical paths by sensor and origin; labels are never arguments to this block.
    unique_sensors=sorted(set(sensors.ravel().tolist()))
    sensor_col={s:cols.index(s) for s in unique_sensors}
    train_origins=_valid_test_origins(frame.timestamp,p1['context_steps'],tr_end,p1['context_steps'],12,p1['sampling_interval_minutes'])
    matcher={s:WindowMatcher(vals[:tr_end,sensor_col[s]],p1['context_steps'],12,train_origins) for s in unique_sensors}
    all_orig=np.concatenate([co,to])
    t0=time.perf_counter(); pilot_n=min(cfg['pilot_queries'],len(unique_sensors)*len(all_orig))
    # Timed exact matrix-vector query pilot using the first sensor and realistic contexts.
    for o in all_orig[:pilot_n]: matcher[unique_sensors[0]].query(vals[o-168:o,sensor_col[unique_sensors[0]]],10)
    elapsed=time.perf_counter()-t0
    projected=elapsed*len(unique_sensors)*len(all_orig)/max(pilot_n,1)
    print(f'kNN timing pilot: {pilot_n} queries in {elapsed:.2f}s; linear projection {projected/60:.1f} min.')
    if projected>7200: raise RuntimeError('Projected kNN run exceeds 2 hours; stop for hosted runtime plan.')
    # Per sample query performed on demand below. The cache avoids repeated matching for the k grid.
    match_cache={}
    for s in unique_sensors:
        for o in all_orig:
            match_cache[(s,int(o))]=matcher[s].query(vals[o-168:o,sensor_col[s]],max(cfg['k_grid']))[0]

    (ROOT/cfg['output_dir']).mkdir(parents=True,exist_ok=True)
    # Calibration-only hyperparameter selection by average 60-minute interval score.
    # Compute q candidates from nearest futures and choose k; no regime labels are used.
    k_scores=[]
    for k in cfg['k_grid']:
        scores=[]
        for g in range(3):
            for si,s in enumerate(sensors[g]):
                for wi,o in enumerate(co):
                    paths=match_cache[(s,int(o))][:k]
                    lo,hi=empirical_interval(paths)
                    scores.append(interval_score(cy[g,si,wi,11],lo[11],hi[11],.1))
        k_scores.append(float(np.mean(scores)))
    best_k=int(cfg['k_grid'][int(np.argmin(k_scores))])
    k_table=pd.DataFrame({'k':cfg['k_grid'],'calibration_interval_score':k_scores})
    k_table.to_csv(ROOT/'outputs/phase2/k_sweep.csv',index=False)

    # Precompute calibration histories and mix candidates for global and paper-style w tuning.
    # Histories and Chronos quantile pseudo-samples are kept separate from scoring labels.
    def method_inputs(forecasts, actuals, origins):
        gN,sN,wN=actuals.shape[:3]
        hist=np.empty((gN,sN,wN,100,12),np.float32); model_s=np.empty_like(hist)
        for g in range(gN):
            for si,s in enumerate(sensors[g]):
                for wi,o in enumerate(origins):
                    hist[g,si,wi]=match_cache[(s,int(o))][:best_k][np.random.default_rng(31+wi).choice(best_k,100,replace=True)]
                    model_s[g,si,wi]=quantiles_to_samples(forecasts[g,si,wi],QUANTILES,100)
        return hist,model_s
    ch,cm=method_inputs(cf,cy,co)
    weight_scores=[]
    for w in cfg['weight_grid']:
        vals_score=[]
        for g in range(3):
            for si in range(30):
                for wi in range(len(co)):
                    sample=mix_samples(cm[g,si,wi],ch[g,si,wi],w,np.random.default_rng(700+wi))
                    lo,hi=np.quantile(sample,[.05,.95],axis=0)
                    vals_score.append(interval_score(cy[g,si,wi,11],lo[11],hi[11],.1))
        weight_scores.append(float(np.mean(vals_score)))
    best_w=float(cfg['weight_grid'][int(np.argmin(weight_scores))])
    pd.DataFrame({'weight':cfg['weight_grid'],'calibration_interval_score':weight_scores}).to_csv(ROOT/'outputs/phase2/bma_weight_sweep_global.csv',index=False)
    # Paper-style approximation: first selected sensor, first 10 calibration origins.
    paper_scores=[]
    for w in cfg['weight_grid']:
        ss=[]
        for wi in range(min(10,len(co))):
            sample=mix_samples(cm[0,0,wi],ch[0,0,wi],w,np.random.default_rng(700+wi))
            lo,hi=np.quantile(sample,[.05,.95],axis=0)
            ss.append(interval_score(cy[0,0,wi,11],lo[11],hi[11],.1))
        paper_scores.append(float(np.mean(ss)))
    paper_w=float(cfg['weight_grid'][int(np.argmin(paper_scores))])
    pd.DataFrame({'weight':cfg['weight_grid'],'calibration_interval_score':paper_scores}).to_csv(ROOT/'outputs/phase2/bma_weight_sweep_paper_style.csv',index=False)

    # Gamma grid evaluated on calibration using one alpha state per sensor/stream shared across horizons.
    gammas=cfg['gamma_grid']
    methods=['ACI approximate nominal 90%','ACI native 80% (P10-P90)',
             'BMA+ACI paper-style weight — our label-free adaptation of BMA, not exact BMA from [1]',
             'BMA+ACI global weight — our label-free adaptation of BMA, not exact BMA from [1]']
    bases={'ACI approximate nominal 90%':(cf,cy,None), 'ACI native 80% (P10-P90)':(cf,cy,None)}
    for name,w in [(methods[2],paper_w),(methods[3],best_w)]: bases[name]=(cf,cy,w)
    gamma_choice={}
    for name,(ff,yy,w) in bases.items():
        target_alpha=.2 if '80%' in name else .1
        vals_by_gamma=[]
        for gamma in gammas:
            losses=[]
            for g in range(3):
                for si in range(30):
                    alpha=target_alpha
                    pending=[]
                    for wi in range(len(co)):
                        # Apply only labels whose target timestamps precede this origin.
                        matured=sorted((event for event in pending if event[0] < co[wi]),key=lambda e:e[0])
                        pending=[event for event in pending if event[0] >= co[wi]]
                        for _,missed in matured: alpha=aci_update(alpha,missed,gamma,target_alpha)
                        if w is None:
                            q=ff[g,si,wi]
                            lo,hi=approximate_90_bounds(q) if alpha==.1 else (q[:,0],q[:,4])
                            center=q[:,2]
                        else:
                            samp=mix_samples(cm[g,si,wi],ch[g,si,wi],w,np.random.default_rng(700+wi))
                            lo,hi=np.quantile(samp,[alpha/2,1-alpha/2],axis=0); center=np.median(samp,axis=0)
                        new_events=[]
                        for h in (2,5,11):
                            l,u=widen_bounds(lo[h],hi[h],alpha,target_alpha)
                            losses.append(float(interval_score(yy[g,si,wi,h],l,u,alpha)))
                            miss=int(yy[g,si,wi,h]<l or yy[g,si,wi,h]>u)
                            new_events.append((int(co[wi])+h+1,miss))
                        pending.extend(new_events)
            vals_by_gamma.append(float(np.mean(losses)))
        gamma_choice[name]=float(gammas[int(np.argmin(vals_by_gamma))])
        pd.DataFrame({'gamma':gammas,'calibration_mean_interval_score':vals_by_gamma}).assign(method=name).to_csv(ROOT/f"outputs/phase2/gamma_sweep_{methods.index(name)}.csv",index=False)

    # Test scoring: all method forecasts and windows share the untouched Phase 1 cache.
    rows=[]; summary=[]
    for method in ['Native Chronos-Bolt (approximate nominal 90%)','ACI approximate nominal 90%',
                   'Native Chronos-Bolt (native 80%)','ACI native 80% (P10-P90)',
                   'History-sampling (our version of [1])',methods[2],methods[3]]:
        for g in range(3):
            for si,s in enumerate(sensors[g]):
                alpha=.1 if '80%' not in method else .2
                gamma=gamma_choice.get(method, gamma_choice.get(methods[0],.005))
                pending=[]
                for wi,o in enumerate(to):
                    matured=sorted((event for event in pending if event[0] < o),key=lambda e:e[0])
                    pending=[event for event in pending if event[0] >= o]
                    target_alpha=.2 if '80%' in method else .1
                    for _,missed in matured: alpha=aci_update(alpha,missed,gamma,target_alpha)
                    f=tf[g,si,wi]; y=ty[g,si,wi]
                    if 'History-sampling' in method:
                        samples=match_cache[(s,int(o))][:best_k]
                        lo,hi=empirical_interval(samples); point=np.median(samples,axis=0)
                    elif 'BMA+ACI' in method:
                        w=paper_w if 'paper-style' in method else best_w
                        hs=match_cache[(s,int(o))][:best_k]
                        hs=hs[np.random.default_rng(31+wi).choice(best_k,100,replace=True)]
                        ms=quantiles_to_samples(f,QUANTILES,100)
                        samples=mix_samples(ms,hs,w,np.random.default_rng(700+wi))
                        lo,hi=np.quantile(samples,[.05,.95],axis=0); point=np.median(samples,axis=0)
                    elif 'native 80%' in method.lower(): lo,hi=f[:,0].copy(),f[:,4].copy(); point=f[:,2].copy()
                    else: lo,hi=approximate_90_bounds(f); point=f[:,2]
                    if method.startswith('ACI') or method.startswith('BMA+ACI'):
                        new_events=[]
                        for h in (2,5,11):
                            l,u=widen_bounds(lo[h],hi[h],alpha,target_alpha)
                            miss=int(y[h]<l or y[h]>u)
                            lo[h],hi[h]=l,u
                            new_events.append((int(o)+h+1,miss))
                        pending.extend(new_events)
                    for h in (2,5,11):
                        label=label_target_period(y[:h+1],p1['free_flow_mph'],p1['congested_mph'])
                        rows.append({'seed':p1['sensor_seeds'][g],'sensor':s,'origin':int(o),'method':method,'horizon_minutes':(h+1)*5,'regime':label,'actual':y[h],'lower':lo[h],'upper':hi[h],'point':point[h],'score':interval_score(y[h],lo[h],hi[h],.2 if '80%' in method.lower() else .1)})
    df=pd.DataFrame(rows); out=ROOT/cfg['output_dir']; out.mkdir(parents=True,exist_ok=True)
    grouped=df.groupby(['method','horizon_minutes','regime'],dropna=False)
    metrics=grouped.apply(lambda x:pd.Series({'n':len(x),'coverage':np.mean((x.actual>=x.lower)&(x.actual<=x.upper)),'mean_width_mph':np.mean(x.upper-x.lower),'interval_score':np.mean(x.score),'point_mae':np.mean(np.abs(x.actual-x.point))}),include_groups=False).reset_index()
    metrics.to_csv(out/'metrics_by_regime.csv',index=False,float_format='%.6f')
    # Headline summary: 60m transition and calm (=free-flow + congested pooled).
    summary=[]
    for method,g in df[(df.horizon_minutes==60)].groupby('method'):
        t=g[g.regime=='transition']; c=g[g.regime!='transition']
        summary.append({'method':method,'transition_n':len(t),'transition_coverage':np.mean((t.actual>=t.lower)&(t.actual<=t.upper)),'calm_n':len(c),'calm_coverage':np.mean((c.actual>=c.lower)&(c.actual<=c.upper)),'mean_width':np.mean(g.upper-g.lower),'interval_score':np.mean(g.score),'mae':np.mean(abs(g.actual-g.point))})
    pd.DataFrame(summary).to_csv(out/'headline.csv',index=False,float_format='%.6f')
    # Two requested figures.
    plot=metrics[metrics.horizon_minutes==60]
    fig,ax=plt.subplots(figsize=(12,7)); pivot=plot.pivot(index='method',columns='regime',values='coverage').reindex(columns=REGIMES)
    pivot.index=[s.replace('Native Chronos-Bolt','Chronos').replace('approximate nominal 90%','approx 90').replace('BMA+ACI','BMA + ACI').replace(' — our label-free adaptation of BMA, not exact BMA from [1]',' (label-free)') for s in pivot.index]
    pivot.plot(kind='barh',ax=ax); ax.axvline(.9,color='k',ls='--',lw=1); ax.set_xlim(0,1); ax.set_xlabel('Coverage (60 minutes)'); ax.set_ylabel(''); ax.grid(axis='x',alpha=.2); ax.legend(title='Outcome regime',loc='lower right'); fig.tight_layout(); fig.savefig(out/'coverage_by_regime.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6)); mp=metrics[(metrics.horizon_minutes==60)&(metrics.regime=='transition')]
    ax.scatter(mp.mean_width_mph,mp.coverage,s=65)
    for _,r in mp.iterrows():
        label=r.method.replace('Native Chronos-Bolt','Chronos').replace('approximate nominal 90%','approx 90').replace(' — our label-free adaptation of BMA, not exact BMA from [1]',' (LF)')
        ax.annotate(label,(r.mean_width_mph,r.coverage),fontsize=7,xytext=(4,3),textcoords='offset points')
    ax.set_xlabel('Mean interval width (mph)'); ax.set_ylabel('Transition coverage (60 minutes)'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'width_vs_coverage.png',dpi=160); plt.close(fig)
    pd.DataFrame([{'k':best_k,'global_weight':best_w,'paper_style_weight':paper_w,**{f'gamma_{i}':v for i,v in enumerate(gamma_choice.values())}}]).to_csv(out/'selected_parameters.csv',index=False)
    print('Headline 60m transition:')
    print(pd.DataFrame(summary)[['method','transition_coverage','transition_n','calm_coverage','mean_width','interval_score','mae']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print(f'Outputs: {out}; calibration cache: {ccpath}; k={best_k}, weights={paper_w:.2f}/{best_w:.2f}; gamma selections saved.')
    _write_report(out,summary,gamma_choice,best_k,paper_w,best_w,cache_status,pilot_n,elapsed,projected)


def _write_report(out,summary,gamma_choice,best_k,paper_w,best_w,cache_status,pilot_n,elapsed,projected):
    lines=['# Phase 2: PEMS-BAY baselines','',
      'All methods share Phase 1 cached test forecasts, sensors and chronological test origins. Regime labels are generated only after predictions and used only for scoring. Calibration tuning uses the preceding chronological block; online ACI updates only after outcomes mature before a later forecast origin. PEMS-BAY (2017) may overlap Chronos training data, so this is a historical reproduction.','',
      f'- Calibration forecast cache status: {cache_status}; 50 sampled calibration origins. Test cache reused unchanged.',
      f'- kNN timing pilot: {pilot_n} exact queries in {elapsed:.3f}s; projected matching {projected:.1f}s. Full run completed in roughly one minute on the host after model load.',
      f'- Calibration selections: k={best_k}, paper-style weight={paper_w:.2f}, global weight={best_w:.2f}. Gamma selections appear in the sweep table.',
      '- Point forecast is the distribution median; ACI changes bounds but not the point forecast. Calm pools free-flow and congested; mean width and interval score pool all regimes at 60 minutes.',
      '- ACI width mapping is an implementation assumption: multiply the symmetric base radius by sqrt(target_alpha/current_alpha), so a smaller online miscoverage level widens the interval. [1] gives no gamma for this experiment.',
      '- Chronos pseudo-samples use deterministic inverse-CDF linear interpolation over q10/q20/q50/q80/q90; tails are held at q10/q90. Mixtures replace round(100*w) model paths with sampled neighbor paths.',
      '- [1] BMA gate uses training-only destination-regime probabilities. This implementation has no regime categories or predicted-regime gate and is explicitly not exact BMA from [1].','',
      '## Headline at 60 minutes','',
      '| Method | Transition coverage (n) | Calm coverage (n) | Mean width | Interval score | Point MAE |','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['transition_coverage']:.3f} ({r['transition_n']}) | {r['calm_coverage']:.3f} ({r['calm_n']}) | {r['mean_width']:.2f} | {r['interval_score']:.2f} | {r['mae']:.2f} |")
    lines += ['', '## Calibration gamma sweep (mean sequential interval score)','',
      '| Method | 0.001 | 0.005 | 0.01 | 0.02 | Selected |','|---|---:|---:|---:|---:|---:|']
    for i in range(4):
        sweep=pd.read_csv(out/f'gamma_sweep_{i}.csv'); name=sweep.method.iloc[0]
        values=' | '.join(f'{v:.2f}' for v in sweep.calibration_mean_interval_score)
        lines.append(f'| {name} | {values} | {gamma_choice[name]:.3f} |')
    # Array summary retains the insertion order: global BMA appears before paper-style.
    lookup={r['method']:r for r in summary}
    hist=lookup['History-sampling (our version of [1])']['transition_coverage']*100
    global_bma=lookup['BMA+ACI global weight — our label-free adaptation of BMA, not exact BMA from [1]']['transition_coverage']*100
    paper_bma=lookup['BMA+ACI paper-style weight — our label-free adaptation of BMA, not exact BMA from [1]']['transition_coverage']*100
    chronos=lookup['Native Chronos-Bolt (approximate nominal 90%)']['transition_coverage']*100
    lines += ['', '## Published comparison (descriptive, not protocol matched)','',
      f'- History-sampling: ours {hist:.1f}% versus [1] about 81–82%, a gap of {hist-81:+.1f} to {hist-82:+.1f} percentage points. [1] does not specify its conditioning estimator or matching details.',
      f'- BMA + ACI: [1] reports 73.6%. Our global label-free adaptation is {global_bma:.1f}% and the paper-style 10-window weight adaptation is {paper_bma:.1f}%; neither is exact BMA because the [1] training-only regime gate is omitted.',
      f'- Native Chronos-Bolt approximate nominal-90 result is {chronos:.1f}% versus [1] 54.9%, a descriptive gap of {chronos-54.9:+.1f} points; our tail extrapolation and window sampling are explicit approximations.',
      '', '## Guesses and deviations','',
      '- History sampler: per-sensor train-only z-scored 168-step contexts, exact Euclidean matching, timestamp-contiguous training origins, k in [5, 10, 20, 50], actual next-hour neighbor paths, and empirical 5th/95th percentile interval. [1] describes current-speed conditioning and 100 draws but omits this matching rule, k, and interval construction.',
      '- Paper-style weight: first sensor in seed-42 group, first 10 sampled calibration origins, and mean 60-minute interval score objective. [1] says 10 held-out windows but leaves sensor/window selection and objective unspecified. Global weight pools all calibration sensors/windows.',
      '- Gamma candidates are [0.001, 0.005, 0.01, 0.02], selected by calibration-only sequential interval score. One alpha state per sensor/method is shared across horizons. Forecasts from one origin share the same state; outcomes update it only when their target timestamps are before a later origin.',
      '- Pseudo-sample interpolation/tail handling and ACI width mapping above are implementation assumptions. Full rows across 15/30/60 minutes, regimes, counts, widths, interval scores and MAE: `metrics_by_regime.csv`. k/weight/gamma sweeps and selected values are adjacent CSVs.',
      '- Plots: `coverage_by_regime.png`, `width_vs_coverage.png`. No official baseline code was found or reused; all baseline logic is implemented locally.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); main(ap.parse_args().force)
