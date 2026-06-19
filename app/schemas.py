from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    data: List[List[float]]
    use_cache: bool = True


class PredictResponse(BaseModel):
    prediction: float
    cache_hit: bool
    latency_ms: float

