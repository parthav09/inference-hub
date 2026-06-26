import time
from collections import OrderedDict

class LRUTTLCache:
    def __init__(self, max_size=256, ttl_seconds = 30):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.enabled = True  # 👈 Agent-controlled switch
    
        self.dicts = OrderedDict()

        self.hits = 0
        self.miss = 0
        self.expire = 0
        self.evict = 0
    

    def get(self, key):
        """Return Cache value if exists, enabled and unexpired"""
        if not self.enabled:
            self.miss += 1
            return None

        if key not in self.dicts:
            self.miss += 1
            return None
    
        value, timestamp = self.dicts[key]

        if time.time() - timestamp>self.ttl:
            self.expire += 1
            del self.dicts[key]
            self.miss += 1
            return None
    
        self.dicts.move_to_end(key)
        self.hits += 1
        return value
    
    def set(self, key, value):
        """Inset into or update the kv store"""

        if not self.enabled:
            return
        
        self.dicts[key] = (value, time.time())
        self.dicts.move_to_end(key)

        if len(self.dicts) > self.max_size:
            self.dicts.popitem(last = False)
            self.evict += 1

    def clear(self):
        """Clear the cache"""
        self.dicts.clear()
        self.hits = 0
        self.miss = 0
        self.expire = 0
        self.evict = 0

    def stats(self):
        total = self.hits + self.miss
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "policy": "LRU with TTL",
            "enabled": self.enabled,
            "size": len(self.dicts),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "miss": self.miss,
            "expire": self.expire,
            "evict": self.evict,
            "hit_rate": round(hit_rate, 3)
        }
