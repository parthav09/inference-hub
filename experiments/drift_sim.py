# experiments/drift_sim.py

import numpy as np

def generate_drifted_features(
    base_prefix=[1,1,1,1],
    step=0,
    drift_strength=0.1
):
    """
    Generates features that gradually drift over time.
    """
    prefix = base_prefix
    suffix = np.random.normal(
        loc=step * drift_strength,
        scale=1.0,
        size=28
    )
    features = prefix + suffix.tolist()
    return features