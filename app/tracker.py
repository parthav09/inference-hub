import time

class TrackPerformance:
    def __init__(self):
        self.records = list()
    
    def record(self, item):
        self.records.append(item)

    def summary(self):
        if not self.records:
            return {}
        
        l = sorted(self.records)
        n = len(l)
        return {
            "count": n,
            "p50": l[int(0.5 * n)],
            "p95": l[int(0.95 * n)],
            "p99": l[int(0.99 * n) if n > 1 else -1],
        }