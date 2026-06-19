import time
import hashlib
import torch
from fastapi import FastAPI
from app.model import InferenceModel
from app.cache import LRUTTLCache
from app.tracker import TrackPerformance
from app.schemas import PredictionRequest, PredictResponse

app = FastAPI()
model = InferenceModel()
cache = LRUTTLCache()
tracker = TrackPerformance()

def make_cache_key(features, k = 4):
    """key is created based on prefix (kv cache logic)"""
    prefix = features[:k]
    embeddings = str(prefix).encode()
    return hashlib.sha256(embeddings).hexdigest()

def agent_policy(risk):
    """Enabling/Disabling cache based on risk"""

    if risk > 0.8:
        cache.enabled = False
    else:
        cache.enabled = True

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictionRequest):
    start = time.time()
    
    x = torch.tensor(request.data).unsqueeze(0).float()
    cache_key = make_cache_key(request.data)
    cache_hit = False

    h = None

    # Try Cache
    if request.use_cache:
        h = cache.get(cache_key)
        if h is not None:
            cache_hit = True
    
    if h is None:
        h = model.get_embedding(x)
        if h is not None:
            cache.set(cache_key, h)
    
    y = model.predict_from_embedding(h)

    risk_score = float(torch.norm(h).item() / 10)

    agent_policy(risk_score)
    latency = (time.time() - start) * 1000

    tracker.record(latency)

    return PredictResponse(
        prediction=float(y.mean().item()),
        cache_hit=cache_hit,
        latency_ms=round(latency, 2),
    )

@app.get("/stats")
def stats():
    return {
        "cache": cache.stats(),
        "latency": tracker.summary(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_enabled": cache.enabled,
    }    





