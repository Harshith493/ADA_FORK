# FORK runbook

## Environment

Windows PowerShell, Python 3.12, CPU. The project uses cached Chronos-Bolt weights under `.cache/huggingface`. Streamlit is installed by the user; this project does not install it or any packages.

For a clean environment, use the dependencies previously used by the project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install chronos-forecasting pandas matplotlib
```

The demo requires Streamlit to be present in the selected environment. Demo startup reads only local caches and result files; it does not load Chronos or access the network.

## Phase commands

Run from the repository root. Chronos model downloads are needed only if model weights are not already cached; the Phase 7 demo does not download anything.

```powershell
python scripts\phase0_forecast.py
python scripts\run_phase1.py
python scripts\run_phase2.py
python scripts\run_phase3.py
python scripts\run_phase4.py
python scripts\run_phase5.py
python scripts\run_phase6.py
python -m unittest discover -s tests -v
```

Phase 6 uses the fixed configuration `configs/phase6.json`; its evaluation reuses Chronos and branch caches. Additional cache-only output utilities:

```powershell
python scripts\run_phase6_bootstrap.py
python scripts\write_phase6_headline.py
python scripts\summarize_phase6.py
python scripts\summarize_phase6_reliability.py
```

The all-origin Phase 6 scale-up is not an executable evaluation yet. `python scripts\run_phase6_scaled.py --estimate-only` reports its approximately 69.5-minute runtime and exits without running it. All-origin forecast/branch cache generation remains to be implemented.

## Demo

Build the ignored local replay cache from existing local PEMS-BAY data and Phase 1/3/4 caches; this performs no inference and no downloads:

```powershell
python scripts\build_phase7_demo_cache.py
streamlit run app\demo.py
```

The smoke test is `python -m unittest tests.test_demo -v`. If the replay cache is missing, the app displays the build command. Preset examples use transition labels for display selection only; the screen explicitly warns about this.

## Main Phase 7 artifacts

- `outputs/phase6/headline_60m_ci.md` - clipped Phase 6 headline table and CIs.
- `outputs/phase6/matched_coverage.csv` and `matched_curves.png` - transition sharpness comparisons.
- `outputs/phase6/matched_calm_coverage.csv` - matched calm-regime comparisons.
- `outputs/phase6/branch_reliability.csv` and `branch_reliability.png` - branch-odds calibration.
- `docs/results_for_report.md` - report-ready summary and claims guidance.
