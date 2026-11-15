"""Cached FORK replay demo; the app performs no model inference or downloads."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data/cache/phase7_demo_cache.npz'

def load_demo_data(path=CACHE):
    """Load prebuilt cache arrays and validate a replay row."""
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    required={'context','quantiles','actual','branch_paths','branch_counts','branch_weights','regime','pflag','training_min','training_max'}
    missing=required-set(data)
    if missing: raise ValueError(f'demo cache missing keys: {sorted(missing)}')
    if len(data['actual'])<1: raise ValueError('demo cache contains no windows')
    return data

def choose_window(data,preset,index=0):
    """Presets use labels only to choose a display example, never as method input."""
    regimes=data['regime'].astype(str); flags=data['pflag'].astype(bool)
    if preset=='Jam onset': candidates=np.flatnonzero(regimes=='transition')
    elif preset=='Free-flow': candidates=np.flatnonzero(regimes=='free-flow')
    elif preset=='False alarm': candidates=np.flatnonzero(flags & (regimes!='transition'))
    else: return int(np.clip(index,0,len(regimes)-1))
    return int(candidates[0]) if len(candidates) else None

def _plot_replay(data,i,reveal):
    fig,ax=plt.subplots(figsize=(9,4)); context=data['context'][i]; q=data['quantiles'][i]; actual=data['actual'][i]
    ax.plot(np.arange(-len(context),0),context,color='#334155',label='Observed context')
    ax.axvline(0,color='#94a3b8',lw=1)
    # Phase 1 quantile order is P10, P20, P50, P80, P90.
    lo=np.clip(q[:,0],data['training_min'][i],data['training_max'][i]); hi=np.clip(q[:,4],data['training_min'][i],data['training_max'][i])
    ax.fill_between(np.arange(1,13),lo,hi,color='#64748b',alpha=.2,label='Chronos P10–P90 (native 80%, physically clipped)')
    ax.plot(np.arange(1,13),q[:,2],color='#475569',lw=1.5,label='Chronos median')
    colors=['#2563eb','#f97316']; paths=data['branch_paths'][i]; counts=data['branch_counts'][i]; weights=data['branch_weights'][i]
    for j in range(2):
        n=int(counts[j])
        if n<1: continue
        sample=paths[j,:n]; lo,hi=np.quantile(sample,[.1,.9],axis=0); lo=np.clip(lo,data['training_min'][i],data['training_max'][i]); hi=np.clip(hi,data['training_min'][i],data['training_max'][i]); mid=np.median(sample,axis=0)
        ax.fill_between(np.arange(1,13),lo,hi,color=colors[j],alpha=.17*float(weights[j])+.06)
        ax.plot(np.arange(1,13),mid,color=colors[j],lw=2,label=f'FORK branch {j+1} (weight {weights[j]:.2f})')
    if reveal: ax.plot(np.arange(1,13),actual,color='#dc2626',lw=2,label='Actual (revealed)')
    ax.set(xlabel='5-minute steps from forecast origin',ylabel='Speed (mph)'); ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); return fig

def main():
    st.set_page_config(page_title='FORK traffic replay',layout='wide'); st.title('FORK · calibrated scenario forecasting')
    st.caption('Cached PEMS-BAY replay only. No model inference or network access occurs in this app.')
    if not CACHE.exists():
        st.error('Replay cache is missing. From the project root, run: python scripts\\build_phase7_demo_cache.py'); return
    try: data=load_demo_data()
    except Exception as exc: st.error(f'Could not load replay cache: {exc}'); return
    replay,coverage,results,limitations=st.tabs(['Replay','Coverage level','Results','Limitations'])
    with replay:
        st.info('Preset selection uses transition labels for display only. Labels never enter FORK, its gates, or its forecasts.')
        preset=st.selectbox('Example window',['Jam onset','Free-flow','False alarm','Index'])
        idx=None
        if preset=='Index': idx=int(st.number_input('Window index',0,len(data['actual'])-1,0,step=1))
        else:
            idx=choose_window(data,preset)
            if idx is None: st.warning(f'No cached window matches preset {preset!r}. Choose Index instead.')
        if idx is not None:
            reveal=st.checkbox('Reveal actual future',value=False)
            st.write(f"Window {idx} · sensor {data['sensors'][idx]} · display regime {data['regime'][idx]} · p(fork) flag {bool(data['pflag'][idx])}")
            st.pyplot(_plot_replay(data,idx,reveal),clear_figure=True)
    with coverage:
        file=ROOT/'outputs/phase6/coverage_width_curve.csv'
        if not file.exists(): st.warning('Phase 6 matched-coverage cache is missing.')
        else:
            frame=pd.read_csv(file); choices=sorted(frame.alpha.unique())
            alpha=st.select_slider('Nominal miscoverage alpha',options=choices,value=.1)
            horizon=st.selectbox('Horizon',sorted(frame.horizon.unique()),index=2,format_func=lambda x:f'{x} minutes')
            def row(method,subset):
                x=frame[(frame.method==method)&(frame.alpha==alpha)&(frame.horizon==horizon)&(frame.subset==subset)]
                return x.iloc[0] if len(x) else None
            methods=['FORK ACI (p(fork))','Chronos scaled ACI (p(fork) gate)']; rows=[]
            for method in methods:
                tr=row(method,'transition'); calm=row(method,'calm-flagged'); fork=row(method,'forked-flagged')
                if tr is not None and calm is not None and fork is not None:
                    rows.append({'Method':method,'Transition coverage':tr.coverage,'Transition width (mph)':tr.width,'Calm group coverage':calm.coverage,'Forked group coverage':fork.coverage})
            if rows: st.dataframe(pd.DataFrame(rows).style.format({'Transition coverage':'{:.1%}','Transition width (mph)':'{:.2f}','Calm group coverage':'{:.1%}','Forked group coverage':'{:.1%}'}),use_container_width=True)
            else: st.warning('This alpha/horizon is not present in the cached Phase 6 results.')
    with results:
        table=ROOT/'outputs/phase6/headline_60m_ci.csv'; curve=ROOT/'outputs/phase6/coverage_width_curve.csv'; plot=ROOT/'outputs/phase6/branch_reliability.png'
        if table.exists(): st.dataframe(pd.read_csv(table),use_container_width=True)
        else: st.warning('Corrected headline table is missing.')
        if curve.exists():
            d=pd.read_csv(curve); d=d[(d.method=='FORK ACI (random)')&(d.alpha==.1)&(d.horizon==60)&d.subset.isin(['calm-flagged','forked-flagged'])]
            if len(d): st.subheader('Random-gate group coverage (physically clipped)'); st.dataframe(d[['subset','coverage','width','n']].style.format({'coverage':'{:.1%}','width':'{:.2f}'}),use_container_width=True)
        if plot.exists(): st.subheader('Branch-odds reliability'); st.image(str(plot))
        else: st.warning('Branch reliability plot is missing.')
    with limitations:
        st.markdown('''
- Transition coverage point estimate and day-block CI are shown in the corrected results table; the target is 85% and the point estimate should be judged directly.
- FORK whole-path region coverage is about 66% because bounds stop at empirical branch support; the 90% whole-path guarantee is not met. Per-step gate-group coverage is near 90%.
- The p(fork) detector kill signal is active: it did not beat volatility or recent-drop in AUPRC.
- PEMS-BAY is from 2017 and may overlap Chronos training data; treat this as a historical reproduction.
- Sensor coupling, other TSFMs, METR-LA transfer, Caltrans 2026, fev-bench and GIFT-Eval are not done.
- Phase 4 branch construction used native P10–P90 (80%) paths; Phase 5/6 calibrate to a 90% target. These are different starting uncertainty levels.
''')

if __name__=='__main__': main()
