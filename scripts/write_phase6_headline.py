"""Build the compact 60-minute FORK-vs-scaled-Chronos table with CIs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'outputs/phase6'
ci=pd.read_csv(out/'day_block_bootstrap_ci.csv'); curves=pd.read_csv(out/'coverage_width_curve.csv')
rows=[]
for method,name,point_metric in [('FORK ACI (p(fork))','FORK ACI (p(fork))','weighted_median_point_mae'),('Chronos scaled ACI (p(fork) gate)','Chronos scaled ACI (p(fork) gate)','chronos_q50_point_mae')]:
 d=ci[ci.method==method].set_index('metric'); sub=curves[(curves.method==method)&(curves.alpha==.1)&(curves.horizon==60)].set_index('subset')
 tr=sub.loc['transition']; calm=sub.loc['calm-flagged']; fork=sub.loc['forked-flagged']
 rows.append({'method':name,'transition_coverage':tr.coverage,'transition_coverage_ci95':f"{d.loc['transition_coverage','ci95_low']:.3f}-{d.loc['transition_coverage','ci95_high']:.3f}",'calm_group_coverage':calm.coverage,'calm_group_ci95':f"{d.loc['calm_group_coverage','ci95_low']:.3f}-{d.loc['calm_group_coverage','ci95_high']:.3f}",'forked_group_coverage':fork.coverage,'forked_group_ci95':f"{d.loc['forked_group_coverage','ci95_low']:.3f}-{d.loc['forked_group_coverage','ci95_high']:.3f}",'transition_mean_width_mph':tr.width,'width_ci95':f"{d.loc['transition_width','ci95_low']:.2f}-{d.loc['transition_width','ci95_high']:.2f}",'transition_interval_score':tr.interval_score,'interval_score_ci95':f"{d.loc['transition_interval_score','ci95_low']:.2f}-{d.loc['transition_interval_score','ci95_high']:.2f}",'point_mae':d.loc[point_metric,'estimate'],'mae_ci95':f"{d.loc[point_metric,'ci95_low']:.2f}-{d.loc[point_metric,'ci95_high']:.2f}",'point_mae_subset':'all test windows at 60m'})
df=pd.DataFrame(rows); df.to_csv(out/'headline_60m_ci.csv',index=False)
cols=list(df.columns); lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
for _,r in df.iterrows(): lines.append('| '+' | '.join(f"{x:.3f}" if isinstance(x,float) else str(x) for x in r)+' |')
(out/'headline_60m_ci.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print((out/'headline_60m_ci.md').read_text())
