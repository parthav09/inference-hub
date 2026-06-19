# experiments/correctness.py

import torch
import torch.nn.functional as F

def cosine_distance(a, b):
    return 1 - F.cosine_similarity(a, b).item()

def l2_distance(a, b):
    return torch.norm(a - b).item()