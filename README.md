# Inference Hub

Inference Hub is an experimental FastAPI service for exploring embedding caching, inference latency, and cache-control policies around a small PyTorch Transformer encoder.

This repository is a prototype, not a production inference system. The model is randomly initialized and untrained, cache state is process-local, and several policy/configuration modules are not yet integrated.

## What the service does

For each `POST /predict` request, the service:

1. Accepts `data` as a sequence of 32-feature rows.
2. Converts it to a tensor with shape `(1, sequence_length, 32)`.
3. Builds a SHA-256 cache key from the first four sequence rows.
4. Looks up a cached pooled embedding when `use_cache` is `true`.
5. On a miss, runs the Transformer backbone and attempts to cache the resulting 96-dimensional embedding.
6. Runs a prediction head that produces 32 sigmoid outputs and returns their mean as one scalar prediction.
7. Computes a risk score from the embedding norm and enables or disables the global cache for subsequent requests.
8. Records request latency in memory.

The service exposes health, prediction, cache statistics, and latency statistics endpoints.

## Important current behavior

The following details are important when interpreting results:

- The model has no trained weights or checkpoint. Predictions are random and change when the process restarts.
- `use_cache=false` skips the cache lookup, but the newly computed embedding is still written to the cache if the global cache is enabled. It is therefore a cache-read bypass, not a strict no-cache mode.
- The cache key uses the first four **sequence rows**, not the first four scalar features. With the one-row requests used by `load_test.py`, the key represents the entire row.
- The load test reuses identical inputs. It does not test reuse across different inputs that merely share a four-value feature prefix.
- For sequences longer than four rows, different sequences with the same first four rows receive the same key even though the cached embedding represents the full sequence. This can return an incorrect embedding.
- The risk score is `norm(embedding) / 10`. Because the backbone ends with a 96-dimensional `LayerNorm`, a typical score is about `0.98`, above the `0.8` threshold. The first request therefore normally disables caching for later requests.
- The policy is applied after lookup, inference, and cache insertion, so a policy decision affects later requests rather than the request that produced the score.
- Cache contents, cache statistics, and latency records are in-memory global state. They reset on restart and are not shared between multiple worker processes.
- The request schema checks that `data` is nested, but does not enforce non-empty input or a row width of 32. Invalid dimensions fail during model execution.

## Project structure

```text
app/
  main.py       FastAPI routes and the active cache policy
  model.py      Transformer encoder and prediction head
  cache.py      In-memory LRU cache with insertion-time TTL
  tracker.py    In-memory latency percentile tracking
  schemas.py    Request and response models
  config.py     Environment configuration declarations (not wired in)

agent/
  policy.py     Standalone policy prototype (not used by the API)
  controller.py Standalone cache controller prototype (not used by the API)

experiments/
  load_test.py               HTTP latency experiments
  plot_day3.py               Benchmark chart generation
  drift_sim.py               Synthetic feature drift helper
  correctness.py             Embedding-distance helpers
  correctness_experiment.py  Stale prototype; currently not runnable

docker/
  Dockerfile
  docker-compose.yml

reports/         Generated benchmark JSON and plots
```

The standalone agent prototype is currently disconnected from `app/main.py`. It also returns lowercase decisions in `agent/policy.py` while `agent/controller.py` expects uppercase decision names, so the two files do not work together without modification.

`experiments/correctness_experiment.py` currently imports a nonexistent `Model` class and calls a nonexistent `encode()` method. It must be updated to use `InferenceModel` before it can run.

## Requirements

- Python 3.10 or newer
- Docker with Docker Compose, if using the container setup

Core dependencies are listed in `requirements.txt`. The plotting and drift helpers also require `matplotlib` and `numpy`, which are not currently listed there.

## Setup

### Docker

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The API is available at [http://localhost:8000](http://localhost:8000).

The Compose file declares `CACHE_MAX_SIZE`, `CACHE_TTL_SECONDS`, `CACHE_ENABLED`, and `PREFIX_K`, but the running application does not currently read those settings. It always starts with the defaults defined directly in `app/cache.py` and `app/main.py`: maximum size `256`, TTL `30` seconds, cache enabled, and four sequence rows in the key.

### Local environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To use the plotting or drift scripts, install their currently unpinned dependencies:

```bash
python -m pip install matplotlib numpy
```

## API

Interactive OpenAPI documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) while the service is running.

### Health

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "cache_enabled": true
}
```

`cache_enabled` will normally become `false` after the first prediction because of the current risk calculation.

### Predict

Each inner array represents one sequence row and must contain exactly 32 numeric features for the current model.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],"use_cache":true}'
```

Response shape:

```json
{
  "prediction": 0.5,
  "cache_hit": false,
  "latency_ms": 12.34
}
```

The numeric values vary by process and machine.

### Statistics

```bash
curl http://localhost:8000/stats
```

The response contains:

- Cache state and counters: `enabled`, `max_size`, `ttl_seconds`, `hits`, `miss`, `expire`, `evict`, and integer percentage `hit_rate`.
- In-process latency statistics: `count`, `p50`, `p95`, and `p99`.

## Running the benchmark

Start the API first, then run:

```bash
python experiments/load_test.py
```

The script sends three workloads to `http://localhost:8000/predict`:

1. 100 identical requests with cache reads bypassed.
2. 100 identical requests with cache reads requested.
3. 250 identical requests with cache reads requested.

All workloads use concurrency `10`. Results are written to:

```text
reports/day3_results.json
```

Because the active risk policy normally disables the cache after the first prediction, the labels `cached_prefix_reuse` and `bursty_cached` do not guarantee cache hits. Check `GET /stats` when evaluating benchmark behavior.

Generate the comparison chart after the JSON report exists:

```bash
python experiments/plot_day3.py
```

Output:

```text
reports/day3_latency_comparison.png
```

The chart compares mean and p95 latency for the first two workloads.

## Highest-priority improvements

- Wire `app/config.py` into cache construction and cache-key generation.
- Decide whether `use_cache=false` should disable both reads and writes.
- Replace the current key with one that matches the embedding being cached, or cache a true reusable prefix state.
- Calibrate and configure the risk policy so cache-enabled experiments can produce actual hits.
- Integrate the agent policy/controller and standardize their decision values.
- Repair the correctness experiment and add its dependencies to `requirements.txt`.
- Add strict request dimension validation, tests, synchronization for shared state, and production observability.

## License

No license file is currently included. Add a license before distributing or accepting external contributions.
