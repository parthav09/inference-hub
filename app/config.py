import os

CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 256))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 30))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

PREFIX_K = int(os.getenv("PREFIX_K", 4))