# Inference Hub

Inference Hub is an experimental inference service built to study one practical question:

> Can reusable model embeddings reduce inference latency without introducing unacceptable correctness risk?

The project combines a small Transformer-based model, an in-memory embedding cache, a policy that can disable caching, and a set of latency experiments. It is intended as a systems prototype for reasoning about the trade-off between speed and correctness—not as a trained prediction product or production-ready serving platform.

## The problem being explored

Model inference often repeats work. If multiple requests contain the same reusable input region, the service may be able to cache an intermediate model representation and skip part of the computation on later requests.

That creates two competing goals:

- **Performance:** reduce repeated computation and improve latency, especially p95 and p99 tail latency.
- **Correctness:** avoid reusing an embedding when the underlying input has changed enough that the cached representation is no longer valid.

Inference Hub makes both sides observable. It records latency and cache behavior while applying a simple risk policy that decides whether the cache should remain enabled.

## System design

The service accepts a sequence in which each row contains 32 numeric features. A request moves through the following pipeline:

```mermaid
flowchart LR
    A["POST /predict"] --> B["Create input tensor"]
    B --> C["Build cache key"]
    C --> D{"Cache read requested?"}
    D -- "Yes" --> E{"Embedding found?"}
    D -- "No" --> F["Run Transformer encoder"]
    E -- "Yes" --> G["Reuse cached embedding"]
    E -- "No" --> F
    F --> H["Attempt cache write"]
    G --> I["Run prediction head"]
    H --> I
    I --> J["Calculate risk score"]
    J --> K["Update cache policy"]
    K --> L["Record latency"]
    L --> M["Return prediction"]
```

### Model

The PyTorch model has two stages:

1. A Transformer encoder converts the input sequence into a pooled 96-dimensional embedding.
2. A prediction head converts that embedding into 32 sigmoid outputs.

The API returns the mean of those 32 outputs as a single prediction.

The model is randomly initialized and has no trained checkpoint. Its purpose is to provide realistic inference computation for cache and latency experiments. The prediction itself has no domain meaning and changes when the process restarts.

### Embedding cache

The cache is an in-memory LRU cache with:

- Maximum size: 256 entries
- TTL: 30 seconds from insertion
- LRU eviction when capacity is exceeded
- Hit, miss, expiration, eviction, and hit-rate counters

The current cache key is a SHA-256 hash of the first four **sequence rows**. This distinction matters: the key does not use the first four scalar features.

The benchmark sends one-row sequences, so its key represents the complete input row. For longer sequences, two inputs with identical first four rows but different later rows receive the same key even though the cached embedding represents the full sequence. That is a known correctness risk in the current prototype.

### Cache-control policy

After inference, the service calculates:

```text
risk_score = embedding_norm / 10
```

If the score is greater than `0.8`, caching is disabled globally for following requests. Otherwise, it is enabled.

Because the 96-dimensional embedding is passed through `LayerNorm`, its norm is typically close to `sqrt(96)`. The resulting risk score is therefore usually around `0.98`, which means the first prediction normally disables the cache.

This is useful as a demonstration of policy-controlled infrastructure, but the threshold is not calibrated well enough for a meaningful cache-performance comparison yet.

## What is measured

The project records two different views of latency.

### Service latency

`POST /predict` measures time inside the API handler. This includes:

- Tensor creation
- Cache lookup
- Model embedding computation on a miss
- Prediction-head computation
- Risk evaluation

The service stores these measurements in memory and exposes:

- **p50:** median request latency
- **p95:** latency below which 95% of requests completed
- **p99:** latency below which 99% of requests completed
- **count:** number of recorded requests

### Client-observed latency

`experiments/load_test.py` measures elapsed time around the complete HTTP request. It therefore includes service execution plus local HTTP and scheduling overhead.

The experiment report contains:

- Mean latency
- p50 latency
- p95 latency
- p99 latency
- Request count

### Cache metrics

`GET /stats` reports:

- Cache hits and misses
- Expired entries
- LRU evictions
- Hit rate as an integer percentage
- Whether the policy currently allows cache use

These metrics are process-local and reset whenever the service restarts.

## Experiment design

The current load test runs three workloads against `POST /predict`:

| Workload | Requests | Concurrency | Input | Cache-read flag |
|---|---:|---:|---|---|
| Baseline | 100 | 10 | Identical one-row sequence | Disabled |
| Cached-prefix reuse | 100 | 10 | Identical one-row sequence | Enabled |
| Larger cached workload | 250 | 10 | Identical one-row sequence | Enabled |

The third workload is named `bursty_cached` in the report, although it increases the request count rather than concurrency.

`use_cache=false` currently bypasses cache reads only. If the global cache is enabled, the newly computed embedding can still be written to the cache. It is therefore not a strict no-cache baseline.

## Recorded latency results

A sample local run generated the following `reports/day3_results.json` measurements:

| Workload | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Baseline, 100 requests | 32.82 ms | 20.55 ms | 146.66 ms | 152.02 ms |
| Cached-prefix reuse, 100 requests | 22.49 ms | 21.27 ms | 35.97 ms | 38.46 ms |
| Larger cached workload, 250 requests | 21.35 ms | 20.90 ms | 25.79 ms | 36.17 ms |

Relative to the recorded baseline, the cached-prefix run shows:

- 31.5% lower mean latency
- 3.5% higher p50 latency
- 75.5% lower p95 latency
- 74.7% lower p99 latency

![Baseline and cached latency comparison](reports/day3_latency_comparison.png)

### How to interpret these results

The lower mean and tail measurements are observations from one local run. They cannot currently be attributed confidently to cache reuse.

The active risk policy normally disables the cache after the first prediction, and the load test does not record whether each request was a cache hit. Warm-up effects, Python scheduling, model initialization, and machine load can therefore explain part or all of the difference.

A defensible cache benchmark should:

1. Use a calibrated or temporarily fixed cache policy.
2. Clear and reset service state before every workload.
3. Make the baseline bypass both cache reads and writes.
4. Record cache-hit status for every request.
5. Run multiple trials and report variation or confidence intervals.
6. Separate warm-up requests from measured requests.
7. Test both exact input reuse and safe prefix reuse.

The current results demonstrate the measurement pipeline, but they should not be treated as a final performance claim.

## Correctness and drift work

The repository includes an early correctness experiment intended to:

1. Generate features that drift over time.
2. Compare a cached embedding with a freshly computed embedding.
3. Measure cosine and L2 distance.
4. Treat embedding divergence as correctness risk.

That is the intended bridge between cache performance and cache safety. However, `experiments/correctness_experiment.py` is currently stale: it imports a nonexistent `Model` class and calls a nonexistent `encode()` method. It must be updated to use `InferenceModel` before this experiment can produce valid results.

## API surface

The service exposes three endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Run inference, optionally attempt cache reuse, and return prediction latency |
| `GET /stats` | Inspect cache counters and latency percentiles |
| `GET /health` | Check service health and whether caching is currently enabled |

A prediction request has this shape:

```json
{
  "data": [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
  "use_cache": true
}
```

Each inner row must contain 32 values. The current schema does not validate that width before model execution.

The response contains:

```json
{
  "prediction": 0.5,
  "cache_hit": false,
  "latency_ms": 12.34
}
```

## Repository map

```text
app/
  main.py       Request flow, cache lookup, active policy, and API routes
  model.py      Transformer encoder and prediction head
  cache.py      In-memory LRU and TTL cache
  tracker.py    In-process latency percentile tracking
  schemas.py    API request and response models
  config.py     Environment settings that are not yet wired into the service

agent/
  policy.py     Standalone policy prototype
  controller.py Standalone cache controller prototype

experiments/
  load_test.py               HTTP latency benchmark
  plot_day3.py               Latency chart generation
  drift_sim.py               Synthetic feature drift
  correctness.py             Embedding-distance functions
  correctness_experiment.py  Incomplete correctness experiment

reports/
  day3_results.json              Generated benchmark data (gitignored)
  day3_latency_comparison.png    Tracked sample chart
```

The standalone files in `agent/` are not used by the API. They also disagree on decision naming: the policy returns lowercase values while the controller expects uppercase values.

The Docker Compose file declares cache environment variables, but `app/main.py` does not currently consume `app/config.py`. Runtime cache settings are therefore still the hard-coded defaults.

## Reproducing the experiment

The service can be started with:

```bash
docker compose -f docker/docker-compose.yml up --build
```

With the API running on port `8000`, generate the measurements and chart with:

```bash
python experiments/load_test.py
python experiments/plot_day3.py
```

The load test requires `httpx`. Plotting additionally requires `matplotlib`; the drift helper requires `numpy`. The latter two are not currently declared in `requirements.txt`.

## Project status

Inference Hub currently proves out the shape of a cache-aware inference experiment:

- A model can expose and reuse intermediate embeddings.
- Cache behavior and latency can be observed through one service.
- A policy can change serving behavior based on a risk signal.
- Performance and correctness experiments can share the same model path.

The next engineering step is to make the experiment scientifically valid: fix cache semantics, calibrate the policy, repair correctness measurement, isolate benchmark runs, and connect cache decisions to measured embedding drift.

## License

No license file is currently included.
