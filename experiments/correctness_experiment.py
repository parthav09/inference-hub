# experiments/correctness_experiment.py
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import time
from app.model import InferenceModel
from experiments.drift_sim import generate_drifted_features
from experiments.correctness import cosine_distance, l2_distance

model = InferenceModel()

def run_correctness_experiment(steps=20):
    cached_embedding = None
    results = []

    for step in range(steps):
        features = generate_drifted_features(step=step)

        x = torch.tensor(features).unsqueeze(0).unsqueeze(0).float()

        fresh_embedding = model.get_embedding(x)

        if cached_embedding is None:
            cached_embedding = fresh_embedding

        cos_dist = cosine_distance(cached_embedding, fresh_embedding)
        l2_dist = l2_distance(cached_embedding, fresh_embedding)
        risk_score = cos_dist  # embedding divergence as correctness risk

        results.append({
            "step": step,
            "cosine_distance": round(cos_dist, 4),
            "l2_distance": round(l2_dist, 4),
            "correctness_risk": round(risk_score, 4)
        })

    return results


if __name__ == "__main__":
    results = run_correctness_experiment()

    for r in results:
        print(r)
