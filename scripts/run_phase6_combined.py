"""Unified Phase‑6 runner with a single APPLY_CLIP flag.
Both runs use identical seeds, caches and code; the only difference is whether
interval endpoints are clipped to the sensor‑wise training min/max.
The script can be invoked with `--clip` (default) or `--no-clip`.
It also re‑uses the gamma values selected in the clipped run when the
unclipped run is executed, ensuring that the only variable is the clipping flag.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_phase5 as p5
from fork.phase1 import _load_period, _split_indices, label_target_period
from fork.phase3 import calendar_features
from fork.phase4 import corr_from_residuals
from fork.phase5 import branch_nonconformity, scaled_chronos_scores
from fork.phase6 import day_block_bootstrap_ci, interpolate_at, minority_reliability, clip_interval

parser = argparse.ArgumentParser()
parser.add_argument('--clip', dest='clip', action='store_true', default=True)
parser.add_argument('--no-clip', dest='clip', action='store_false')
parser.add_argument('--fixed-gamma', type=str, default=None,
                    help='Path to a CSV with selected gammas to enforce (used when --no-clip)')
args = parser.parse_args()
APPLY_CLIP = args.clip
FIXED_GAMMA_PATH = args.fixed_gamma

def maybe_clip(sim, ix, h, low, high):
    if APPLY_CLIP:
        lo, hi = clip_interval(sim['lower'][ix, h], sim['upper'][ix, h], low, high)
        lo = np.atleast_1d(lo)
        hi = np.atleast_1d(hi)
        return lo, hi
    else:
        return np.array([sim['lower'][ix, h]]), np.array([sim['upper'][ix, h]])

def main():
    t0 = time.perf_counter()
    p6 = json.loads((ROOT / 'configs' / 'phase6.json').read_text())
    cfg = json.loads((ROOT / p6['phase5_config']).read_text())
    p1 = json.loads((ROOT / cfg['phase1_config']).read_text())
    p3 = json.loads((ROOT / cfg['phase3_config']).read_text())
    out = ROOT / p6['output_dir']
    out.mkdir(parents=True, exist_ok=True)

    print('Loading cached arrays and labels for scoring only...', flush=True)
    frame, columns = _load_period(ROOT / p1['dataset_path'], p1['period_start'], p1['period_end_exclusive'])
    values = frame[columns].to_numpy(np.float32)
    train_end, cal_end = _split_indices(len(frame), p1['train_fraction'], p1['calibration_fraction'])

    with np.load(ROOT / 'data' / 'cache' / 'phase1_pems_bay_forecasts.npz') as z:
        tq, ty, to, sensors = z['forecasts'], z['actuals'], z['origins'], z['sensors']
    with np.load(ROOT / 'data' / 'cache' / 'phase2_calib_forecasts.npz') as z:
        cq, cy, co = z['forecasts'], z['actuals'], z['origins']
    with np.load(ROOT / p3['train_forecast_cache']) as z:
        pooled = corr_from_residuals(z['actuals'] - z['forecasts'][:, :, 2])
    p4 = json.loads((ROOT / cfg['phase4_config']).read_text())
    with np.load(ROOT / p4['branch_cache']) as z:
        tp, tc, tw = z['branch_paths'], z['branch_counts'], z['weights']
    with np.load(ROOT / cfg['branch_cache']) as z:
        cp, cc, cw = z['cal_paths'], z['cal_counts'], z['cal_weights']

    cq0, cy0, co0, cg, csi = p5.flatten_inputs(cq, cy, co, sensors)
    tq0, ty0, to0, tg, tsi = p5.flatten_inputs(tq, ty, to, sensors)
    ncal = len(cy0)
    q = np.concatenate([cq0, tq0])
    y = np.concatenate([cy0, ty0])
    origins = np.concatenate([co0, to0])
    groups = np.concatenate([cg, tg])
    sensor_slots = np.concatenate([csi, tsi])
    paths = np.concatenate([cp, tp])
    counts = np.concatenate([cc, tc])
    weights = np.concatenate([cw, tw])
    ns = len(y)

    # build per‑sensor training limits
    column_index = {str(name): i for i, name in enumerate(columns)}
    speed_limits = {}
    for g in range(sensors.shape[0]):
        for s in range(sensors.shape[1]):
            name = str(sensors[g, s])
            col = column_index[name]
            speed_limits[(g, s)] = (float(values[:train_end, col].min()),
                                    float(values[:train_end, col].max()))

    def clipped_objective(sim, actual, group_ids, slots, alpha):
        losses = []
        for h in p5.HORIZONS:
            los = np.empty(len(actual), float)
            his = np.empty(len(actual), float)
            for ix, (g, s) in enumerate(zip(group_ids, slots)):
                low, high = speed_limits[(int(g), int(s))]
                lo, hi = maybe_clip(sim, ix, h, low, high)
                los[ix] = lo
                his[ix] = hi
            losses.extend(p5.interval_score(actual[:, h], los, his, alpha).tolist())
        return float(np.mean(losses))

    print('Computing cached-path scores...', flush=True)
    sorted_paths = np.sort(paths, axis=2)
    fork_scores = p5.score_paths(sorted_paths, counts, weights, y, .1, 20)
    calmw = np.zeros_like(weights)
    anchors = np.zeros(ns, dtype=np.int8)
    for i in range(ns):
        ds = [np.linalg.norm(np.nanmedian(paths[i, j, :counts[i, j]], axis=0) - q[i, :, 2])
              if counts[i, j] else np.inf for j in (0, 1)]
        anchors[i] = int(np.argmin(ds))
        calmw[i, anchors[i]] = 1
    calm_scores = p5.score_paths(sorted_paths, counts, calmw, y, .1, 20)
    chron = np.stack([p5.sample_chronos(a, 5505 + i) for i, a in enumerate(q)])
    chron_sort = np.sort(chron, axis=1)
    chron_score = np.stack([branch_nonconformity(chron_sort[i][None], [200], [1.], y[i], .1, 20, True)
                            for i in range(ns)])
    scaled = np.stack([scaled_chronos_scores(a, b) for a, b in zip(q, y)])

    print('Tuning calibration-only gammas and replaying test...', flush=True)
    with np.load(ROOT / p3['feature_cache']) as z:
        cb, tb = z['cal_base'], z['test_base']
    with np.load(ROOT / p3['fork_output_cache']) as z:
        pf = np.concatenate([z['p_fork_cal'] > .168745, z['p_fork_test'] > .168745])
    rate = float(pf[:ncal].mean())
    threshold = float(np.quantile(cb[:, 1], 1 - rate, method='higher'))
    recent = np.concatenate([cb[:, 1] >= threshold, tb[:, 1] >= threshold])
    def random_gate(n, seed):
        rng = np.random.default_rng(seed)
        flag = np.zeros(n, bool)
        flag[rng.permutation(n)[:int(round(n * rate))]] = True
        return flag
    gates = {
        'p(fork)': pf,
        'recent-drop': recent,
        'random': np.concatenate([random_gate(ncal, cfg['seed']),
                                   random_gate(len(ty0), cfg['seed'] + 1)])
    }
    gammas = cfg['gamma_grid']
    rows = []
    raw = []
    chosen = []
    base_cfg = dict(cfg)
    base_cfg['target_alpha'] = .1

    # Load fixed gammas if supplied (used for the no‑clip run)
    fixed_gammas = {}
    if FIXED_GAMMA_PATH:
        df_fixed = pd.read_csv(FIXED_GAMMA_PATH)
        for _, r in df_fixed.iterrows():
            fixed_gammas[(r['alpha'], r['method'])] = r['gamma']

    for alpha in p6['alpha_targets']:
        specs = [
            ('FORK ACI (p(fork))', 'fork', 'p(fork)', gammas),
            ('Chronos scaled ACI (p(fork) gate)', 'chronos_scaled', 'p(fork)', gammas)
        ]
        if float(alpha) == .1:
            specs.append(('FORK ACI (random)', 'fork', 'random', [.005]))
        for method, kind, gate_name, method_gammas in specs:
            # use fixed gamma if provided
            if (alpha, method) in fixed_gammas:
                gamma = fixed_gammas[(alpha, method)]
                chosen.append({'alpha': alpha, 'method': method, 'gamma': gamma,
                               'gamma_selection': 'fixed from clipped run'})
            else:
                sweep = []
                tune_cfg = dict(base_cfg)
                tune_cfg['target_alpha'] = alpha
                for gamma in method_gammas:
                    sim = p5.simulate(origins[:ncal], groups[:ncal], y[:ncal],
                                      sorted_paths[:ncal], counts[:ncal], weights[:ncal],
                                      anchors[:ncal], chron_sort[:ncal], q[:ncal],
                                      fork_scores[:ncal], calm_scores[:ncal],
                                      chron_score[:ncal], scaled[:ncal],
                                      gates[gate_name][:ncal], kind, gamma, tune_cfg)
                    sweep.append((clipped_objective(sim, y[:ncal], groups[:ncal], sensor_slots[:ncal], alpha), gamma))
                gamma = min(sweep)[1]
                chosen.append({'alpha': alpha, 'method': method, 'gamma': gamma,
                               'gamma_selection': 'calibration‑only clipped interval score'})
            # run full replay with selected gamma
            sim = p5.simulate(origins, groups, y, sorted_paths, counts, weights,
                              anchors, chron_sort, q, fork_scores, calm_scores,
                              chron_score, scaled, gates[gate_name], kind, gamma, base_cfg)
            for h in (2, 5, 11):  # 15,30,60 minutes
                for i in range(ncal, ns):
                    # Retrieve per‑sensor limits safely
                    key = (int(groups[i]), int(sensor_slots[i]))
                    if key in speed_limits:
                        low, high = speed_limits[key]
                    else:
                        # fallback: compute from raw values using column name
                        name = str(sensors[int(groups[i]), int(sensor_slots[i])])
                        col = column_index[name]
                        low = float(values[:train_end, col].min())
                        high = float(values[:train_end, col].max())
                    regime = label_target_period(y[i, :h+1], p1['free_flow_mph'], p1['congested_mph'])
                    calm = not sim['group'][i]
                    actual = float(y[i, h])
                    lo, hi = maybe_clip(sim, i, h, low, high)
                    width = float(hi[0] - lo[0])
                    score = float(p5.interval_score(np.array([actual]), lo, hi, alpha)[0])
                    covered = float(lo[0] <= actual <= hi[0])
                    rows.append({
                        'method': method,
                        'alpha': alpha,
                        'horizon': (h+1)*5,
                        'regime': regime,
                        'calm': calm,
                        'covered': covered,
                        'width': width,
                        'interval_score': score
                    })
                    if alpha == .1 and method in ('FORK ACI (p(fork))', 'Chronos scaled ACI (p(fork) gate)') and h == 11:
                        raw.append(rows[-1])

    print(f'Calibration sweep and test replay: {(time.perf_counter()-t0)/60:.1f} minutes.', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f'matched_curves_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)

    # 60‑min summary for printing
    tr = df[(df.horizon == 60) & (df.regime == 'transition')]
    for method in ['FORK ACI (p(fork))', 'Chronos scaled ACI (p(fork) gate)']:
        m = tr[tr.method == method]
        print(f"{method} transition width 60m: {m.width.mean():.6f} interval score 60m: {m.interval_score.mean():.6f}")

    # Coverage‑width curves (same as original script)
    curve = []
    for (method, alpha, h), g in df.groupby(['method', 'alpha', 'horizon']):
        partitions = [
            (g[g.regime == 'transition'], 'transition'),
            (g[g.regime != 'transition'], 'non-transition'),
            (g[g.calm], 'calm-flagged'),
            (g[~g.calm], 'forked-flagged')
        ]
        for subset, name in partitions:
            if len(subset):
                curve.append({
                    'method': method, 'alpha': alpha, 'horizon': h,
                    'subset': name,
                    'coverage': subset.covered.mean(),
                    'width': subset.width.mean(),
                    'interval_score': subset.interval_score.mean(),
                    'n': len(subset)
                })
    curve_df = pd.DataFrame(curve)
    curve_df.to_csv(out / f'coverage_width_curve_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)

    # Matched coverage (transition)
    match = []
    for h in (15, 30, 60):
        a = curve_df[(curve_df.horizon == h) & (curve_df.subset == 'transition')]
        for target in p6['matched_transition_coverages']:
            vals = {}
            for method, g in a.groupby('method'):
                vals[method] = (interpolate_at(g.coverage, g.width, target),
                                interpolate_at(g.coverage, g.interval_score, target))
            fw, fi = vals.get('FORK ACI (p(fork))', (None, None))
            cw, ci = vals.get('Chronos scaled ACI (p(fork) gate)', (None, None))
            match.append({
                'horizon': h,
                'target_transition_coverage': target,
                'fork_width': fw,
                'chronos_width': cw,
                'width_reduction': (cw - fw) / cw if cw else None,
                'fork_interval_score': fi,
                'chronos_interval_score': ci
            })
    pd.DataFrame(match).to_csv(out / f'matched_coverage_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)

    # Matched calm coverage (90%)
    calm_match = []
    for h in (15, 30, 60):
        a = curve_df[(curve_df.horizon == h) & (curve_df.subset == 'non-transition')]
        target = p6['matched_calm_coverage']
        vals = {}
        for method, g in a.groupby('method'):
            vals[method] = (interpolate_at(g.coverage, g.width, target),
                            interpolate_at(g.coverage, g.interval_score, target))
        fw, fi = vals.get('FORK ACI (p(fork))', (None, None))
        cw, ci = vals.get('Chronos scaled ACI (p(fork) gate)', (None, None))
        calm_match.append({
            'horizon': h,
            'target_calm_coverage': target,
            'fork_width': fw,
            'chronos_width': cw,
            'width_reduction': (cw - fw) / cw if cw else None,
            'fork_interval_score': fi,
            'chronos_interval_score': ci
        })
    pd.DataFrame(calm_match).to_csv(out / f'matched_calm_coverage_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)

    # Bootstrap CIs (same as original script, using raw records from 60‑min replay)
    main = pd.DataFrame(raw)
    timestamps = frame.timestamp
    day_lookup = {int(o): pd.Timestamp(timestamps.iloc[int(o)]).date().toordinal() for o in origins}
    main['day'] = main.origin.map(day_lookup)
    unique_days = sorted(main.day.unique())
    day_idx = {d: i for i, d in enumerate(unique_days)}
    ids = main.day.map(day_idx).to_numpy()
    specs = {}
    for metric, mask in [
        ('transition_coverage', main.regime.eq('transition')),
        ('calm_group_coverage', main.calm),
        ('forked_group_coverage', ~main.calm),
        ('transition_width', main.regime.eq('transition')),
        ('transition_interval_score', main.regime.eq('transition'))
    ]:
        val = main.covered if metric.endswith('coverage') else main.width if metric.endswith('width') else main.interval_score
        specs[metric] = (mask.to_numpy(), val.to_numpy())
    specs['weighted_median_point_mae'] = (np.ones(len(main), bool), main.fork_abs_error.to_numpy())
    specs['chronos_q50_point_mae'] = (np.isfinite(main.chronos_abs_error.to_numpy()), main.chronos_abs_error.to_numpy())
    cis = []
    for method, g in main.groupby('method'):
        loc = g.index.to_numpy()
        local_ids = ids[loc]
        block = {k: (m[loc], v[loc]) for k, (m, v) in specs.items()}
        ci = day_block_bootstrap_ci(local_ids, block, len(unique_days), p6['bootstrap_resamples'], p6['bootstrap_seed'])
        for metric, (lo, hi) in ci.items():
            subset = 'all test windows at 60m' if 'point_mae' in metric else 'transition windows' if metric.startswith('transition_') else 'gate-flagged group'
            cis.append({
                'method': method,
                'metric': metric,
                'subset': subset,
                'estimate': float(np.nanmean(block[metric][1][block[metric][0]])),
                'ci95_low': lo,
                'ci95_high': hi
            })
    pd.DataFrame(cis).to_csv(out / f'day_block_bootstrap_ci_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)

    # Compact headline table (same as original script)
    ci_df = pd.DataFrame(cis)
    table = []
    for method, point_metric in [
        ('FORK ACI (p(fork))', 'weighted_median_point_mae'),
        ('Chronos scaled ACI (p(fork) gate)', 'chronos_q50_point_mae')
    ]:
        d = ci_df[ci_df.method == method].set_index('metric')
        m = main[main.method == method]
        tr = m[m.regime == 'transition']
        calm = m[m.calm]
        fork = m[~m.calm]
        table.append({
            'method': method,
            'transition_coverage': tr.covered.mean(),
            'transition_coverage_ci95': f"{d.loc['transition_coverage','ci95_low']:.3f}-{d.loc['transition_coverage','ci95_high']:.3f}",
            'calm_group_coverage': calm.covered.mean(),
            'calm_group_ci95': f"{d.loc['calm_group_coverage','ci95_low']:.3f}-{d.loc['calm_group_coverage','ci95_high']:.3f}",
            'forked_group_coverage': fork.covered.mean(),
            'forked_group_ci95': f"{d.loc['forked_group_coverage','ci95_low']:.3f}-{d.loc['forked_group_coverage','ci95_high']:.3f}",
            'transition_width_mph': tr.width.mean(),
            'width_ci95': f"{d.loc['transition_width','ci95_low']:.2f}-{d.loc['transition_width','ci95_high']:.2f}",
            'interval_score': tr.interval_score.mean(),
            'interval_score_ci95': f"{d.loc['transition_interval_score','ci95_low']:.2f}-{d.loc['transition_interval_score','ci95_high']:.2f}",
            'point_mae': d.loc[point_metric,'estimate'],
            'mae_ci95': f"{d.loc[point_metric,'ci95_low']:.2f}-{d.loc[point_metric,'ci95_high']:.2f}",
            'point_mae_subset': 'all test windows at 60m'
        })
    pd.DataFrame(table).to_csv(out / f'headline_60m_ci_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)
    # markdown version for completeness
    cols = list(table[0].keys())
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---']*len(cols)) + ' |']
    for r in table:
        lines.append('| ' + ' | '.join(str(v) if not isinstance(v, float) else f"{v:.3f}" for v in r.values()) + ' |')
    (out / f'headline_60m_ci_{"clip" if APPLY_CLIP else "noclip"}.md').write_text('\n'.join(lines)+'\n')

    pd.DataFrame(chosen).to_csv(out / f'chosen_gamma_{"clip" if APPLY_CLIP else "noclip"}.csv', index=False)
    # after building df
    # Compute raw vs clipped width means for assertion
    raw_widths = []
    clipped_widths = []
    # iterate again through all windows and horizons to capture raw and clipped widths
    for i in range(ncal, ns):
        for h in (2,5,11):
            # Retrieve per‑sensor limits safely (same logic as above)
            key = (int(groups[i]), int(sensor_slots[i]))
            if key in speed_limits:
                low, high = speed_limits[key]
            else:
                name = str(sensors[int(groups[i]), int(sensor_slots[i])])
                col = column_index[name]
                low = float(values[:train_end, col].min())
                high = float(values[:train_end, col].max())
            # raw interval (no clipping)
            raw_lo = sim['lower'][i, h]
            raw_hi = sim['upper'][i, h]
            raw_w = float(raw_hi - raw_lo)
            raw_widths.append(raw_w)
            # clipped interval per current APPLY_CLIP setting
            lo, hi = maybe_clip(sim, i, h, low, high)
            clipped_w = float(hi[0] - lo[0])
            clipped_widths.append(clipped_w)
    raw_mean = np.mean(raw_widths) if raw_widths else float('nan')
    clipped_mean = np.mean(clipped_widths) if clipped_widths else float('nan')
    print('APPLY_CLIP =', APPLY_CLIP)
    print('Mean raw width (all scored windows):', raw_mean)
    print('Mean clipped width (all scored windows):', clipped_mean)
    assert raw_mean >= clipped_mean or not APPLY_CLIP, 'Clipping increased mean width!'


if __name__ == '__main__':
    main()
