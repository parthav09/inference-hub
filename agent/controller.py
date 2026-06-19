# app/agent/controller.py

class Controller:
    """
    Applies agent decisions by adjusting cache configuration.
    """

    def __init__(self, cache, default_ttl=30):
        self.cache = cache
        self.default_ttl = default_ttl

    def apply(self, decision: str):
        if decision == "DISABLE_CACHE":
            self.cache.enabled = False

        elif decision == "SHORT_TTL":
            self.cache.enabled = True
            self.cache.ttl = 5

        else:
            self.cache.enabled = True
            self.cache.ttl = self.default_ttl