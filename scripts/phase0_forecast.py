"""Run one CPU-only Chronos-Bolt forecast on the PEMS-BAY CSV."""

from pathlib import Path
import os
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from chronos import BaseChronosPipeline


DATA = ROOT / "data" / "raw" / "PEMS-BAY.csv"
OUTPUT = ROOT / "outputs" / "phase0" / "chronos_bolt_pems_bay.png"
MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT_STEPS = 14 * 12  # 14 hours at 5-minute intervals, matching the paper.
HORIZON_STEPS = 12  # 1 hour ahead.


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Expected public dataset at {DATA}")

    torch.set_num_threads(4)
    frame = pd.read_csv(DATA, usecols=[0, 1], parse_dates=[0])
    frame = frame.rename(columns={frame.columns[0]: "timestamp", frame.columns[1]: "speed_mph"})
    split = int(len(frame) * 0.7)
    start = split - CONTEXT_STEPS
    stop = split + HORIZON_STEPS
    context = frame.loc[start:split - 1, "speed_mph"].astype("float32").to_numpy()
    actual = frame.loc[split:stop - 1, "speed_mph"].astype("float32").to_numpy()

    pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu")
    quantile_levels = [0.1, 0.5, 0.9]
    model_input = torch.tensor(context)

    def predict():
        return pipeline.predict_quantiles(
            inputs=model_input,
            prediction_length=HORIZON_STEPS,
            quantile_levels=quantile_levels,
        )

    predict()  # warm-up; excluded from timing
    cpu_seconds = []
    wall_seconds = []
    forecast = None
    for _ in range(10):
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        forecast, _ = predict()
        wall_seconds.append(time.perf_counter() - wall_start)
        cpu_seconds.append(time.process_time() - cpu_start)
    assert forecast is not None
    quantiles = forecast[0].detach().cpu().numpy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    hist_times = frame.loc[start:split - 1, "timestamp"]
    future_times = frame.loc[split:stop - 1, "timestamp"]
    plt.figure(figsize=(11, 5))
    plt.plot(hist_times, context, color="#516273", linewidth=1, label="14-hour context")
    plt.fill_between(future_times, quantiles[:, 0], quantiles[:, 2], color="#6fa8dc", alpha=0.3, label="80% interval (10th–90th quantiles)")
    plt.plot(future_times, quantiles[:, 1], color="#1155cc", marker="o", label="Median forecast")
    plt.plot(future_times, actual, color="#cc4125", marker="o", label="Observed (sanity check)")
    plt.axvline(future_times.iloc[0], color="#777777", linestyle="--", linewidth=1)
    plt.title("Chronos-Bolt Small: one-hour PEMS-BAY forecast (sensor 400001)")
    plt.ylabel("Speed (mph)")
    plt.xlabel("Time")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=160)
    print(f"model={MODEL_ID} device=cpu sensor=400001")
    print(f"context={hist_times.iloc[0]} to {hist_times.iloc[-1]} ({CONTEXT_STEPS} points)")
    print(f"forecast={future_times.iloc[0]} to {future_times.iloc[-1]} ({HORIZON_STEPS} points)")
    print(f"median_first={quantiles[0, 1]:.2f} mph actual_first={actual[0]:.2f} mph")
    print(f"timed_repeats=10 median_cpu={statistics.median(cpu_seconds):.4f}s median_wall={statistics.median(wall_seconds):.4f}s")
    print(f"plot={OUTPUT}")
    print("Note: this is an inference sanity check, not a benchmark result.")


if __name__ == "__main__":
    main()
