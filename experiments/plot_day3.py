# experiments/plot_day3.py
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_FILE = REPORTS_DIR / "day3_results.json"
PLOT_FILE = REPORTS_DIR / "day3_latency_comparison.png"

with RESULTS_FILE.open(encoding="utf-8") as f:
    results = json.load(f)

labels = ["Mean", "P95"]
baseline = [
    results["baseline_no_cache"]["mean_latency_ms"],
    results["baseline_no_cache"]["p95"]
]
cached = [
    results["cached_prefix_reuse"]["mean_latency_ms"],
    results["cached_prefix_reuse"]["p95"]
]

x = range(len(labels))
plt.bar(x, baseline, width=0.4, label="No Cache")
plt.bar([i + 0.4 for i in x], cached, width=0.4, label="Cached")

plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel("Latency (ms)")
plt.title("Day 3: Baseline vs Cached Latency")
plt.legend()

plt.savefig(PLOT_FILE)
plt.show()