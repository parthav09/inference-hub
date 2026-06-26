import asyncio
import time
import httpx
import statistics
import json
from pathlib import Path

URL = "http://localhost:8000/predict"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
REPORT_FILE = REPORTS_DIR / "day3_results.json"

FEATURE_A = [1, 1, 1, 1] +[0] * 28
FEATURE_B = [1, 1, 1, 1] +[9] * 28

async def send_request(client, features, use_cache = True):
    start = time.time()
    response = await client.post(URL, json = {
        "data": [features],
        "use_cache": use_cache
    }
    )
    response.raise_for_status()
    end = time.time()
    return (end - start)*1000

async def run_load_test(
        num_requests = 100,
        concurrency = 10,
        reuse_prefix = True,
        use_cache = True
):
    latencies = []

    # simulating concurrent request
    async with httpx.AsyncClient(timeout=30.0) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def task(i):
            async with semaphore:
                features = FEATURE_A if reuse_prefix else FEATURE_B
                latency = await send_request(client, features, use_cache)
                latencies.append(latency)
            
        await asyncio.gather(*(task(i) for i in range(num_requests)))
    
    return latencies


def summary(latencies):
    p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) > 1 else latencies[0]
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
    return {
        "count": len(latencies),
        "mean_latency_ms": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p95": p95,
        "p99": p99,
    }



if __name__ == "__main__":
    print("Load all experiments")

    # Experiment 1: Baseline without Cache
    print("Experiment 1: Baseline without Cache")

    baseline_latencies = asyncio.run(run_load_test(
        num_requests = 100,
        concurrency = 10,
        reuse_prefix = True,
        use_cache = False
    ))
    baseline_summary = summary(baseline_latencies)
    print(summary(baseline_latencies))


    # Experiment 2:With cache
    print("Experiment 2: With Cache")

    with_cache_latencies = asyncio.run(run_load_test(
        num_requests = 100,
        concurrency = 10,
        reuse_prefix = True,
        use_cache = True
    ))
    with_cache_summary = summary(with_cache_latencies)
    print(summary(with_cache_latencies))

    #Experiment 3: Burst traffic
    print("Experiment 3: Burst Traffic")

    burst_latency = asyncio.run(run_load_test(
        num_requests = 250,
        concurrency = 50,
        reuse_prefix = True,
        use_cache = True
    ))
    burst_summary = summary(burst_latency)
    print(summary(burst_latency))

    # -----------------------------
    # Save results
    # -----------------------------
    results = {
        "baseline_no_cache": baseline_summary,
        "cached_prefix_reuse": with_cache_summary,
        "bursty_cached": burst_summary,
    }

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {REPORT_FILE}")
