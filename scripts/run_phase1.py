"""Run the reproducible Phase 1 PEMS-BAY baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT / "src"))

from fork.phase1 import METHOD_APPROX_90, METHOD_NATIVE_80, run_phase1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the matching forecast cache")
    args = parser.parse_args()
    result = run_phase1(ROOT / "configs" / "phase1_pems_bay.json", ROOT, force=args.force)
    metrics = result["metrics"]
    headline = metrics.loc[
        (metrics["seed"] == "all")
        & (metrics["horizon_steps"] == 12)
        & (metrics["regime"] == "transition")
    ].set_index("interval")
    approx = headline.loc[METHOD_APPROX_90]
    native = headline.loc[METHOD_NATIVE_80]
    print(f"Phase 1 done | cache={result['cache_status']} | Jan-May rows={result['frame_rows']:,}")
    print(
        f"60 min transition | {METHOD_APPROX_90}: {approx.coverage:.3f} (n={int(approx.n)})"
    )
    print(f"60 min transition | {METHOD_NATIVE_80}: {native.coverage:.3f} (n={int(native.n)})")
    print(f"metrics={result['metrics_path']}")
    print(f"plot={result['plot_path']}")
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()
