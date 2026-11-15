"""Estimate the all-origin Phase 6 run before its intentionally long execution."""
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/phase6_scaled.json'); ap.add_argument('--estimate-only',action='store_true'); a=ap.parse_args()
 cfg=json.loads((ROOT/a.config).read_text()); p1=json.loads((ROOT/cfg['phase1_config']).read_text())
 from sys import path; path.insert(0,str(ROOT/'src'))
 from fork.phase1 import _load_period,_split_indices
 frame,_=_load_period(ROOT/p1['dataset_path'],p1['period_start'],p1['period_end_exclusive']); train_end,cal_end=_split_indices(len(frame),p1['train_fraction'],p1['calibration_fraction'])
 all_test_origins=np.arange(cal_end, len(frame)-12, cfg['stride_origins'])
 n=int(len(all_test_origins)*cfg['sensor_count']); sampled=4500
 estimate=cfg['estimated_sample_run_minutes']*n/sampled
 print(f"Test origins per sensor: {len(all_test_origins)} at stride {cfg['stride_origins']} (30 minutes).")
 print(f"Windows: {n:,}; proportional A-D estimate: {estimate:.1f} min; branch/forecast cache creation adds overhead.")
 if estimate>30 or a.estimate_only:
  print('Not run. This scale-up exceeds the 30-minute limit and requires all-origin forecast and branch caches; the reviewed 4,500-window outputs remain the measured Phase 6 result.')
  return
 raise SystemExit('No all-origin forecast/branch cache is present. Generate and validate it before running A-D; see HANDOVER.md.')
if __name__=='__main__': main()
