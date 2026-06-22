# Inference Hub

Inference Hub is a small experimental lab for serving model inference with cache-aware behavior and measuring performance under different request patterns.

It combines:
- A FastAPI inference API
- A lightweight Transformer-style PyTorch model
- An LRU + TTL embedding cache
- A simple risk-based policy that can disable cache
- Experiment scripts to benchmark latency and generate reports

---

## What This Project Is

This project is an experimentation sandbox for one question:

How much can prefix-based embedding caching improve inference latency, and what trade-offs appear when we apply policy-based cache control?

The API receives feature vectors, computes embeddings and predictions, optionally reuses cached embeddings, and tracks latency statistics.

---

## Why This Project Exists

Modern inference systems need to balance:
- Speed (lower latency)
- Efficiency (fewer repeated computations)
- Correctness and robustness (avoid stale or risky cache use)

Inference Hub is designed to make these trade-offs easy to explore with controlled experiments.

---

## Goal and Aim

### Goal
Build a practical prototype for testing cache-policy decisions in an inference service.

### Aim
Run repeatable experiments that compare:
- No-cache baseline
- Cache-enabled behavior
- Bursty traffic behavior

Then store and visualize results in the reports directory.

---

## When To Use This Project

Use this project when you want to:
- Learn how inference-time caching impacts p50, p95, and p99 latency
- Prototype policy logic that dynamically enables/disables cache
- Build benchmark reports before moving to larger production-grade systems

---

## How It Works

### Request Flow
1. Client sends a POST request to /predict with data and use_cache.
2. Service builds a cache key from the input prefix.
3. If cache is enabled and key exists, cached embedding is used.
4. Otherwise, model embedding is computed and cached.
5. Prediction is produced from embedding.
6. Risk score is computed and policy may disable cache.
7. Latency is recorded and returned.

### Core Components
- app/main.py: API routes, cache lookup, policy application, response
- app/model.py: Transformer backbone + prediction head
- app/cache.py: LRU cache with TTL and stats
- app/tracker.py: latency percentile tracking
- app/schemas.py: request/response contracts
- experiments/load_test.py: latency benchmark experiments
- experiments/plot_day3.py: chart generation from report JSON

---

## Project Structure

- app: API, model, cache, tracker, schemas
- agent: policy and controller prototypes
- experiments: benchmarking and plotting scripts
- reports: generated JSON and plots
- docker: Dockerfile and Compose configuration

---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyTorch
- Pydantic
- httpx
- Docker + Docker Compose

---

## Setup

### Option A: Run with Docker (recommended)

From repository root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Service runs on http://localhost:8000.

### Option B: Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data":[[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],"use_cache":true}'
```

### Stats

```bash
curl http://localhost:8000/stats
```

---

## Running Experiments

Make sure API is running first.

### 1) Generate benchmark report

```bash
python experiments/load_test.py
```

Output:
- reports/day3_results.json

### 2) Generate latency comparison chart

```bash
python experiments/plot_day3.py
```

Output:
- reports/day3_latency_comparison.png

---

## Notes and Current Behavior

- The cache policy currently disables cache when risk_score > 0.8.
- Depending on model output scale, cache may become disabled quickly.
- If cache_enabled becomes false in /health or /stats, repeated requests can still show cache_hit as false.

---

## Future Improvements

- Add configurable policy thresholds via environment variables
- Add richer experiment suites for drift and correctness
- Add unit and integration tests
- Add throughput metrics and request tracing
- Add CI for linting, tests, and experiment validation

---

## License

Add your preferred license file (for example, MIT) and update this section.
